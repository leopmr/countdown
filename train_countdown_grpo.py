"""
Entraînement GRPO sur la tâche Countdown, avec la librairie TRL.

Version 3 : récompense à trois paliers et plus de puzzles par mise à jour.

Ce qui change par rapport à la version précédente, et pourquoi :
  - Récompense en trois paliers (format 0.1, validité 0.1, correction 1.0) au lieu de deux.
    Le diagnostic d'évaluation montrait que près de la moitié des sorties échouaient avant
    la comparaison de valeur (caractères interdits, nombres inventés) sans que rien ne les
    distingue d'une expression valide mais fausse.
  - Trois fonctions de récompense séparées : TRL journalise la moyenne de chacune
    ("rewards/<nom>/mean"), ce qui permet de voir quel palier progresse.
  - 4 puzzles par mise à jour au lieu de 1 (gradient moins bruité).
  - Plus de repetition_penalty (elle pénalise aussi les tokens du prompt, et les boucles
    qu'elle visait ont disparu avec le chat template).
  - L'évaluation n'est plus faite ici : elle passe par eval_countdown.py, avec un jeu fixe
    exclu de l'entraînement. L'évaluation intégrée au Trainer coûtait du GPU pour une mesure
    différente de celle qu'on utilise pour comparer les modèles.

Prérequis (sur Colab) :
    pip install trl transformers datasets accelerate peft

Dans un notebook :
    from train_countdown_grpo import build_hf_dataset, make_config, REWARD_FUNCS, peft_config, MODEL_NAME
    config = make_config(output_dir="/content/drive/MyDrive/countdown-grpo-v3", max_steps=60)
    trainer = GRPOTrainer(model=MODEL_NAME, reward_funcs=REWARD_FUNCS, args=config,
                          train_dataset=build_hf_dataset(), peft_config=peft_config)
    trainer.train()

En ligne de commande :
    python train_countdown_grpo.py
"""

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from countdown_grpo_starter import (
    REWARD_WEIGHTS,
    completion_text,
    build_prompt,
    generate_dataset,
    reward_correctness,
    reward_format,
    reward_valid,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./countdown-grpo-qwen1.5b-lora-v3"


# ---------------------------------------------------------------------------
# 1. Dataset au format conversationnel (TRL applique le chat template)
# ---------------------------------------------------------------------------
# Les paramètres par défaut doivent rester alignés avec TRAIN_KWARGS, TRAIN_SEED et
# TRAIN_N_PUZZLES de eval_countdown.py : c'est ce qui permet d'exclure les puzzles
# d'entraînement du jeu d'évaluation.

def build_hf_dataset(n_puzzles=2000, n_numbers=3, max_val=10, target_range=(2, 100), seed=0):
    puzzles = generate_dataset(
        n_puzzles=n_puzzles, n_numbers=n_numbers, max_val=max_val,
        target_range=target_range, seed=seed,
    )
    records = [
        {
            "prompt": [{"role": "user", "content": build_prompt(p)}],
            "numbers": p["numbers"],
            "target": p["target"],
        }
        for p in puzzles
    ]
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# 2. Trois fonctions de récompense (une par palier)
# ---------------------------------------------------------------------------
# Signature imposée par GRPOTrainer : (prompts, completions, **kwargs) -> list[float].
# Les colonnes du dataset ("numbers", "target") arrivent dans kwargs. Le nom de la fonction
# sert de clé dans les logs : rewards/format_reward/mean, etc. Une sortie inattendue du
# modèle ne doit jamais interrompre l'entraînement, d'où les try/except.

def format_reward(prompts, completions, **kwargs):
    out = []
    for c in completions:
        try:
            out.append(reward_format(completion_text(c)))
        except Exception:
            out.append(0.0)
    return out


def valid_reward(prompts, completions, numbers, **kwargs):
    out = []
    for c, nums in zip(completions, numbers):
        try:
            out.append(reward_valid(completion_text(c), nums))
        except Exception:
            out.append(0.0)
    return out


def correct_reward(prompts, completions, numbers, target, **kwargs):
    out = []
    for c, nums, tgt in zip(completions, numbers, target):
        try:
            out.append(reward_correctness(completion_text(c), nums, tgt))
        except Exception:
            out.append(0.0)
    return out


REWARD_FUNCS = [format_reward, valid_reward, correct_reward]


def countdown_reward_func(prompts, completions, numbers, target, **kwargs):
    """Somme pondérée des trois paliers en une seule fonction (même valeur que le total
    calculé par TRL avec reward_weights). Utile pour un test rapide en notebook, mais on perd
    le suivi séparé de chaque palier dans les logs : préfère REWARD_FUNCS."""
    f = format_reward(prompts, completions)
    v = valid_reward(prompts, completions, numbers)
    c = correct_reward(prompts, completions, numbers, target)
    wf, wv, wc = REWARD_WEIGHTS
    return [wf * a + wv * b + wc * d for a, b, d in zip(f, v, c)]


# ---------------------------------------------------------------------------
# 3. LoRA
# ---------------------------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


# ---------------------------------------------------------------------------
# 4. Configuration
# ---------------------------------------------------------------------------

def make_config(**overrides):
    """Configuration de référence. Chaque valeur est explicite (pas de dépendance à des défauts
    de version). Passe des surcharges par mots-clés : make_config(max_steps=100, beta=0.0)."""
    params = dict(
        output_dir=OUTPUT_DIR,
        num_generations=8,                  # taille du groupe GRPO (G)
        per_device_train_batch_size=4,      # micro-batch (mémoire)
        gradient_accumulation_steps=8,      # batch effectif 4 x 8 = 32 complétions = 4 puzzles
                                            # (doit rester un multiple de num_generations)
        learning_rate=1e-5,
        beta=0.04,                          # coefficient KL
        temperature=1.0,                    # défaut TRL, inchangé pour isoler l'effet de la récompense
        max_completion_length=200,
        max_steps=60,                       # 60 pas x 4 puzzles = 240 puzzles vus
        reward_weights=list(REWARD_WEIGHTS),
        logging_steps=1,
        save_steps=20,                      # checkpoints fréquents (Drive) : Colab peut se déconnecter
        eval_strategy="no",                 # l'évaluation passe par eval_countdown.py
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16"},
    )
    params.update(overrides)
    return GRPOConfig(**params)


def main():
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=REWARD_FUNCS,
        args=make_config(),
        train_dataset=build_hf_dataset(),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()