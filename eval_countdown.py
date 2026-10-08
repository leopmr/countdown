"""
Évaluation standardisée d'un modèle Countdown (modèle de base ou base + LoRA).

Trois modes :
  short : 50 puzzles, génération déterministe. Contrôle rapide pendant les itérations.
  mid   : 300 puzzles, génération déterministe seule. Suffisant pour la comparaison appariée
          entre deux modèles (voir compare_evals.py), nettement moins long que "long".
  long  : 300 puzzles déterministes, + échantillonnage (pass@k), + puzzles plus durs
          (test de généralisation). Pour les résultats du rapport.

Chaque résultat contient le détail par puzzle (juste ou non, étape d'échec, et la réponse
complète en mode déterministe), ce qui permet la comparaison appariée de compare_evals.py.

Exemples :
  python eval_countdown.py --mode short
  python eval_countdown.py --mode short --adapter /content/drive/MyDrive/.../checkpoint-100 --tag run1-step100
  python eval_countdown.py --mode long  --adapter /content/drive/MyDrive/.../checkpoint-100 --tag run1-step100

Sans --adapter, on évalue le modèle de base : c'est la baseline à comparer.

Dans un notebook, pour éviter de recharger le modèle :
  from eval_countdown import run_eval, load_model
  model, tok = load_model("Qwen/Qwen2.5-1.5B-Instruct", adapter_path)
  run_eval(model, tok, mode="short", tag="mon-test")

Ce qui rend les comparaisons valables :
  - le jeu d'évaluation est fixe (graine fixe) et son empreinte (hash) est enregistrée
    dans chaque résultat : deux résultats ne se comparent que si l'empreinte est identique ;
  - les puzzles d'évaluation sont dédupliqués et exclus du jeu d'entraînement, sinon
    on mesure de la mémorisation. L'espace de puzzles est petit (environ 2500 puzzles
    distincts en 3 nombres entre 1 et 10), donc ce risque est réel ;
  - chaque résultat est accompagné d'un intervalle de confiance à 95 %.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from countdown_grpo_starter import (
    ALLOWED_CHARS,
    _uses_numbers_correctly,
    build_prompt,
    extract_answer,
    generate_dataset,
    reward_correctness,
    reward_format,
)

# --- paramètres du protocole (à ne changer que volontairement : ça change le jeu d'éval) ---
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_SEED = 12345            # graine du jeu d'évaluation
HARD_SEED = 54321            # graine du jeu de puzzles plus durs
TRAIN_SEED = 0               # graine utilisée pour générer le jeu d'entraînement
TRAIN_N_PUZZLES = 2000       # taille du jeu d'entraînement généré (build_hf_dataset)
TRAIN_KWARGS = dict(n_numbers=3, max_val=10, target_range=(2, 100))   # idem entraînement

MODES = {
    "short": dict(n_eval=50, n_sampled=0, k=0, n_hard=0),
    "mid": dict(n_eval=300, n_sampled=0, k=0, n_hard=0),
    "long": dict(n_eval=300, n_sampled=100, k=4, n_hard=100),
}


# ---------------------------------------------------------------------------
# Jeux d'évaluation
# ---------------------------------------------------------------------------

def _key(p):
    return (tuple(sorted(p["numbers"])), p["target"])


def build_eval_set(n, seed=EVAL_SEED, exclude_train=True, **gen_kwargs):
    """n puzzles distincts, déterministes, hors jeu d'entraînement. Le jeu 'short' est
    un préfixe du jeu 'long' : mêmes graine, même ordre."""
    gen = {**TRAIN_KWARGS, **gen_kwargs}
    excluded = set()
    if exclude_train:
        train = generate_dataset(n_puzzles=TRAIN_N_PUZZLES, seed=TRAIN_SEED, **TRAIN_KWARGS)
        excluded = {_key(p) for p in train}

    pool = generate_dataset(n_puzzles=max(20 * n, 2000), seed=seed, **gen)
    out, seen = [], set()
    for p in pool:
        k = _key(p)
        if k in excluded or k in seen:
            continue
        seen.add(k)
        out.append(p)
        if len(out) == n:
            return out
    raise RuntimeError(
        f"Seulement {len(out)} puzzles distincts disponibles hors entraînement (demandé : {n}). "
        "L'espace de puzzles est trop petit pour cette taille d'évaluation."
    )


def fingerprint(puzzles):
    blob = json.dumps([(p["numbers"], p["target"]) for p in puzzles], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Modèle et génération
# ---------------------------------------------------------------------------

def load_model(base_model_name=BASE_MODEL, adapter_path=None, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _eos_ids(model, tokenizer):
    ids = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def generate_batch(model, tokenizer, prompts, max_new_tokens, do_sample=False,
                   temperature=0.7, num_return_sequences=1):
    """Génère pour une liste de prompts (messages de chat). Retourne, pour chaque
    séquence générée, (texte, nb de tokens, tronqué ?). Les séquences d'un même prompt
    sont consécutives quand num_return_sequences > 1."""
    import torch

    chats = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in prompts]
    tokenizer.padding_side = "left"
    inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      num_return_sequences=num_return_sequences,
                      pad_token_id=tokenizer.pad_token_id)
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    eos = _eos_ids(model, tokenizer)
    results = []
    for row in out:
        gen = row[inputs["input_ids"].shape[1]:].tolist()
        n_tok, truncated = len(gen), True
        for i, t in enumerate(gen):
            if t in eos:
                n_tok, truncated = i, False
                break
        text = tokenizer.decode(gen[:n_tok], skip_special_tokens=True)
        results.append((text, n_tok, truncated))
    return results


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def wilson(successes, n, z=1.96):
    """Intervalle de confiance à 95 % (méthode de Wilson) pour une proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def stage(text, numbers, target):
    expr = extract_answer(text)
    if expr is None:
        return "1. format cassé"
    if not ALLOWED_CHARS.match(expr):
        return "2. caractères interdits"
    if not _uses_numbers_correctly(expr, numbers):
        return "3. mauvais nombres"
    return "5. juste" if reward_correctness(text, numbers, target) == 1.0 else "4. mauvaise valeur"


def bootstrap_ci(values, n_boot=5000, seed=0):
    """IC95 par bootstrap en rééchantillonnant les PUZZLES (pas les tirages) : les k tirages
    d'un même puzzle sont corrélés, les traiter comme indépendants donnerait un intervalle
    trop étroit."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def summarize(samples, puzzles, k=1, keep_text=False):
    """samples : liste de (texte, n_tokens, tronqué), k séquences par puzzle consécutives.
    Retourne (résumé, exemples). Le résumé contient "per_puzzle", le détail de chaque puzzle."""
    n_p = len(puzzles)
    correct = [[] for _ in range(n_p)]
    stage_of = [[] for _ in range(n_p)]
    texts = [[] for _ in range(n_p)]
    stages, fmt, lengths, trunc = Counter(), 0, [], 0
    examples = {}
    for i, (text, n_tok, truncated) in enumerate(samples):
        j = i // k
        p = puzzles[j]
        ok = reward_correctness(text, p["numbers"], p["target"]) == 1.0
        s = stage(text, p["numbers"], p["target"])
        correct[j].append(ok)
        stage_of[j].append(s)
        if keep_text:
            texts[j].append(text)
        stages[s] += 1
        examples.setdefault(s, text[-300:])
        fmt += reward_format(text) == 1.0
        lengths.append(n_tok)
        trunc += truncated

    n_s = len(samples)
    n_ok = sum(sum(c) for c in correct)
    if k == 1:
        ci = wilson(n_ok, n_s)
    else:
        ci = bootstrap_ci([sum(c) / len(c) for c in correct])

    per_puzzle = []
    for j, p in enumerate(puzzles):
        row = {"numbers": p["numbers"], "target": p["target"],
               "correct": correct[j], "stage": stage_of[j]}
        if keep_text:
            row["completion"] = texts[j][0] if k == 1 else texts[j]
        per_puzzle.append(row)

    summary = {
        "n_puzzles": n_p,
        "n_samples": n_s,
        "accuracy": n_ok / n_s,
        "ci95": [ci[0], ci[1]],
        "format_rate": fmt / n_s,
        "mean_tokens": sum(lengths) / n_s,
        "truncated_rate": trunc / n_s,
        "stages": dict(sorted(stages.items())),
        "per_puzzle": per_puzzle,
    }
    if k > 1:
        summary["pass_at_k"] = sum(any(c) for c in correct) / n_p
        summary["k"] = k
    return summary, examples


def _run_batches(model, tokenizer, puzzles, max_new_tokens, batch_size, **gen_kwargs):
    samples = []
    nrs = gen_kwargs.get("num_return_sequences", 1)
    step = max(1, batch_size // nrs)
    for i in range(0, len(puzzles), step):
        chunk = puzzles[i:i + step]
        prompts = [[{"role": "user", "content": build_prompt(p)}] for p in chunk]
        samples += generate_batch(model, tokenizer, prompts, max_new_tokens, **gen_kwargs)
    return samples


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, mode="short", tag="", adapter=None, base_model=BASE_MODEL,
             max_new_tokens=200, batch_size=16, out_dir="eval_results"):
    cfg = MODES[mode]
    t0 = time.time()
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "tag": tag, "base_model": base_model,
        "adapter": adapter, "max_new_tokens": max_new_tokens,
    }

    eval_set = build_eval_set(cfg["n_eval"])
    result["eval_set_hash"] = fingerprint(eval_set)

    print(f"\n[1/..] Évaluation déterministe sur {len(eval_set)} puzzles")
    greedy = _run_batches(model, tokenizer, eval_set, max_new_tokens, batch_size, do_sample=False)
    result["greedy"], examples = summarize(greedy, eval_set, keep_text=True)

    if cfg["n_sampled"]:
        sub = eval_set[:cfg["n_sampled"]]
        print(f"[2/..] Échantillonnage : {len(sub)} puzzles x {cfg['k']} tirages (T=0.7)")
        sampled = _run_batches(model, tokenizer, sub, max_new_tokens, batch_size,
                               do_sample=True, temperature=0.7, num_return_sequences=cfg["k"])
        result["sampled"], _ = summarize(sampled, sub, k=cfg["k"])

    if cfg["n_hard"]:
        hard = build_eval_set(cfg["n_hard"], seed=HARD_SEED, exclude_train=False,
                              n_numbers=4, max_val=25, target_range=(10, 999))
        print(f"[3/..] Puzzles plus durs (4 nombres, cible jusqu'à 999) : {len(hard)} puzzles")
        hard_s = _run_batches(model, tokenizer, hard, max_new_tokens, batch_size, do_sample=False)
        result["hard"], _ = summarize(hard_s, hard)
        result["hard_set_hash"] = fingerprint(hard)

    result["duration_s"] = round(time.time() - t0)
    print_report(result, examples)
    save(result, out_dir)
    return result


def print_report(r, examples=None):
    g = r["greedy"]
    print("\n" + "=" * 64)
    print(f"mode={r['mode']}  tag={r['tag'] or '-'}  adapter={r['adapter'] or 'aucun (base)'}")
    print(f"jeu d'éval : {g['n_puzzles']} puzzles, empreinte {r['eval_set_hash']}")
    print("=" * 64)

    def line(name, s):
        lo, hi = s["ci95"]
        extra = f"  pass@{s['k']}={s['pass_at_k']:.1%}" if "pass_at_k" in s else ""
        print(f"{name:12} précision {s['accuracy']:6.1%}  [IC95 {lo:.1%} à {hi:.1%}]  "
              f"format {s['format_rate']:.0%}  tronquées {s['truncated_rate']:.0%}  "
              f"long. moy. {s['mean_tokens']:.0f} tok{extra}")

    line("déterministe", g)
    if "sampled" in r:
        line("échantillonné", r["sampled"])
    if "hard" in r:
        line("puzzles durs", r["hard"])
    print("\nRépartition des échecs (déterministe) :")
    for st, c in g["stages"].items():
        print(f"  {st:26} {c:4d}  ({c / g['n_samples']:.0%})")
    print(f"\ndurée : {r['duration_s']} s")


def save(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"].replace(":", "").replace("-", "")
    name = f"{stamp}_{result['mode']}_{result['tag'] or 'run'}.json"
    (out / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: result[k] for k in ("timestamp", "mode", "tag", "adapter", "eval_set_hash")}
    summary["accuracy"] = round(result["greedy"]["accuracy"], 4)
    summary["ci95"] = [round(x, 4) for x in result["greedy"]["ci95"]]
    with open(out / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"\nrésultats : {out / name}\nhistorique : {out / 'history.jsonl'}")


def main():
    ap = argparse.ArgumentParser(description="Évaluation standardisée Countdown")
    ap.add_argument("--mode", choices=list(MODES), default="short")
    ap.add_argument("--adapter", default=None, help="dossier d'un checkpoint LoRA ; omis = modèle de base")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--tag", default="", help="étiquette libre, ex. run1-step100")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="eval_results")
    a = ap.parse_args()

    model, tokenizer = load_model(a.base_model, a.adapter)
    run_eval(model, tokenizer, mode=a.mode, tag=a.tag, adapter=a.adapter, base_model=a.base_model,
             max_new_tokens=a.max_new_tokens, batch_size=a.batch_size, out_dir=a.out_dir)


if __name__ == "__main__":
    main()  