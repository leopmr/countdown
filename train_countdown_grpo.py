"""
Entraînement GRPO sur la tâche Countdown, avec la librairie TRL.

Version 2 : Qwen2.5-1.5B-Instruct + LoRA, suite aux observations sur
Qwen2.5-0.5B-Instruct (dérive du modèle, format peu respecté, coût
mémoire du modèle de référence KL).

Pourquoi LoRA ici : en passant peft_config à GRPOTrainer, TRL n'a plus
besoin de charger une copie séparée du modèle de référence pour le
calcul de la pénalité KL — il réutilise le même modèle avec l'adaptateur
LoRA désactivé. Ça évite de doubler la mémoire GPU, ce qui a causé le
OutOfMemoryError observé avec un ref_model séparé.

Prérequis (sur Colab, PAS besoin ici pour lire le script) :
    pip install trl transformers datasets accelerate peft

Lancer avec :
    python train_countdown_grpo.py

Ce script suppose que countdown_grpo_starter.py est dans le même
dossier (il réutilise generate_dataset, build_prompt, reward_format,
reward_correctness définis là-bas).
"""

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from countdown_grpo_starter import (
    generate_dataset,
    build_prompt,
    reward_format,
    reward_correctness,
    completion_text,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./countdown-grpo-qwen1.5b-lora"


# ---------------------------------------------------------------------------
# 1. Construction du dataset au format attendu par GRPOTrainer
# ---------------------------------------------------------------------------
# GRPOTrainer attend une colonne "prompt". Les autres colonnes (ici
# "numbers" et "target") sont transmises telles quelles à la reward
# function via **kwargs, une valeur par exemple du dataset — c'est pour
# ça qu'on les garde à côté du prompt plutôt que de les cacher dedans.

def build_hf_dataset(n_puzzles=2000, n_numbers=3, max_val=10, target_range=(2, 100), seed=0):
    puzzles = generate_dataset(
        n_puzzles=n_puzzles, n_numbers=n_numbers, max_val=max_val,
        target_range=target_range, seed=seed,
    )
    records = [
        {
            "prompt": [{"role": "user", "content": build_prompt(p)}],  # format conversationnel : TRL applique le chat template
            "numbers": p["numbers"],
            "target": p["target"],
        }
        for p in puzzles
    ]
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# 2. Reward function au format attendu par GRPOTrainer
# ---------------------------------------------------------------------------
# Signature imposée : (prompts, completions, **kwargs) -> list[float].
# "numbers" et "target" arrivent dans kwargs car ce sont des colonnes du
# dataset, répétées automatiquement par le trainer pour chaque complétion
# générée à partir du même prompt (num_generations complétions par
# prompt, donc par exemple du dataset).

def countdown_reward_func(prompts, completions, numbers, target, **kwargs):
    rewards = []
    for completion, nums, tgt in zip(completions, numbers, target):
        completion = completion_text(completion)
        try:
            r = 0.1 * reward_format(completion) + 1.0 * reward_correctness(completion, nums, tgt)
        except Exception:
            r = 0.0  # une sortie inattendue du modèle ne doit jamais interrompre l'entraînement
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# 3. Configuration LoRA
# ---------------------------------------------------------------------------
# r et lora_alpha modestes, suffisants pour ce genre de tâche sans
# alourdir l'entraînement. target_modules="all-linear" applique LoRA à
# toutes les couches linéaires du modèle plutôt que de lister
# manuellement q_proj/k_proj/v_proj/o_proj, plus simple et généralement
# aussi performant pour un premier essai.

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


# ---------------------------------------------------------------------------
# 4. Évaluation simple : taux de réussite sur un jeu de puzzles tenus à part
# ---------------------------------------------------------------------------
# Utile pour suivre la vraie métrique qui t'intéresse (pas seulement la
# reward moyenne pendant l'entraînement) : le pourcentage de puzzles
# résolus correctement, à comparer avant/après entraînement.

def evaluate_success_rate(model, tokenizer, eval_dataset, max_new_tokens=256):
    import torch

    model.eval()
    successes = 0
    for example in eval_dataset:
        text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        successes += reward_correctness(completion, example["numbers"], example["target"])
    return successes / len(eval_dataset)


# ---------------------------------------------------------------------------
# 5. Entraînement
# ---------------------------------------------------------------------------

def main():
    dataset = build_hf_dataset(n_puzzles=2000)
    split = dataset.train_test_split(test_size=0.05, seed=0)
    train_dataset, eval_dataset = split["train"], split["test"]

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=8,                 # taille du groupe GRPO (G)
        per_device_train_batch_size=4,      # petit mini-batch pour tenir sur une T4
        gradient_accumulation_steps=2,      # batch effectif 8 = num_generations (un groupe par mise à jour)
        learning_rate=1e-5,                 # un peu plus élevé qu'en full fine-tuning, usage courant avec LoRA
        beta=0.04,                          # coefficient KL par rapport au modèle de référence
        max_completion_length=256,          # augmenté après avoir observé un clipped_ratio de 1.0 à 180 tokens
        repetition_penalty=1.15,            # limite les boucles de tokens répétés observées sur le 0.5B
        temperature=0.8,                    # légèrement réduit par rapport à 1.0 pour limiter la dérive incohérente
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        report_to="none",                   # mets "wandb" si tu veux le suivi en ligne
        model_init_kwargs={"dtype": "bfloat16"},   # "torch_dtype" n'était pas pris en compte (modèle chargé en float32)
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=countdown_reward_func,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    print("Évaluation avant entraînement...")
    success_before = evaluate_success_rate(
        trainer.model, trainer.processing_class, eval_dataset
    )
    print(f"Taux de réussite avant : {success_before:.2%}")

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    print("Évaluation après entraînement...")
    success_after = evaluate_success_rate(
        trainer.model, trainer.processing_class, eval_dataset
    )
    print(f"Taux de réussite après : {success_after:.2%}")


if __name__ == "__main__":
    main()