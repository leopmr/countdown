"""
Entraînement GRPO sur la tâche Countdown, avec la librairie TRL.

Prérequis (sur Colab ou ta machine d'entraînement, PAS besoin ici pour
lire le script) :
    pip install trl transformers datasets accelerate

Lancer avec :
    python train_countdown_grpo.py

Ce script suppose que countdown_grpo_starter.py est dans le même
dossier (il réutilise generate_dataset, build_prompt, reward_format,
reward_correctness définis là-bas).
"""

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from countdown_grpo_starter import (
    generate_dataset,
    build_prompt,
    reward_format,
    reward_correctness,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./countdown-grpo-qwen0.5b"


# ---------------------------------------------------------------------------
# 1. Construction du dataset au format attendu par GRPOTrainer
# ---------------------------------------------------------------------------
# GRPOTrainer attend une colonne "prompt". Les autres colonnes (ici
# "numbers" et "target") sont transmises telles quelles à la reward
# function via **kwargs, une valeur par exemple du dataset — c'est pour
# ça qu'on les garde à côté du prompt plutôt que de les cacher dedans.

def build_hf_dataset(n_puzzles=2000, n_numbers=4, seed=0):
    puzzles = generate_dataset(n_puzzles=n_puzzles, n_numbers=n_numbers, seed=seed)
    records = [
        {
            "prompt": build_prompt(p),
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
        r = 0.1 * reward_format(completion) + 1.0 * reward_correctness(completion, nums, tgt)
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# 3. Évaluation simple : taux de réussite sur un jeu de puzzles tenus à part
# ---------------------------------------------------------------------------
# Utile pour suivre la vraie métrique qui t'intéresse (pas seulement la
# reward moyenne pendant l'entraînement) : le pourcentage de puzzles
# résolus correctement, à comparer avant/après entraînement.

def evaluate_success_rate(model, tokenizer, eval_dataset, max_new_tokens=256):
    import torch

    model.eval()
    successes = 0
    for example in eval_dataset:
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)
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
# 4. Entraînement
# ---------------------------------------------------------------------------

def main():
    dataset = build_hf_dataset(n_puzzles=2000)
    split = dataset.train_test_split(test_size=0.05, seed=0)
    train_dataset, eval_dataset = split["train"], split["test"]

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=8,                 # taille du groupe GRPO (G)
        per_device_train_batch_size=8,      # doit être un multiple de num_generations
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        beta=0.04,                          # coefficient KL par rapport au modèle de référence
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        report_to="none",                   # mets "wandb" si tu veux le suivi en ligne
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=countdown_reward_func,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
