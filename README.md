# Countdown with GRPO

GRPO post-training of Qwen2.5-1.5B-Instruct (LoRA) on a Countdown arithmetic task, with a paired evaluation protocol. After 150 steps, greedy accuracy goes from 10.7% to 29.3% on 300 held-out puzzles (paired exact test, p < 1e-7). The full method, results, statistics and limitations are in the report: [`countdown_grpo_report.pdf`](countdown_grpo_report.pdf).

This README explains how the code and files are organised and how to run each part.

## Repository layout

```
.
|-- countdown_grpo_starter.py   puzzles, prompt, reward functions (no training or model code)
|-- train_countdown_grpo.py     GRPO training: dataset, reward wrappers, LoRA and trainer config
|-- eval_countdown.py           standardised evaluation (fixed sets, greedy and sampled, failure stages)
|-- compare_evals.py            paired statistical comparison of two evaluation files
|-- null_model.py               exact accuracy of a random valid expression
|-- countdown_grpo.ipynb        Colab notebook: training, evaluation, comparisons (run in order)
|-- countdown_grpo_report.pdf   report: method, results, limitations
|-- README.md                   this file
`-- results/                    created by eval_countdown.py (not versioned by default)
    |-- <timestamp>_<mode>_<tag>.json   one file per evaluation run
    `-- history.jsonl                   one summary line appended per run
```

## How the pieces depend on each other

```
countdown_grpo_starter.py
   ^            ^
   |            |
train_countdown_grpo.py     eval_countdown.py  <---  null_model.py
                                  |
                           results/*.json
                                  |
                           compare_evals.py
```

- `countdown_grpo_starter.py` is the base layer. It imports nothing from the project, so rewards and puzzles behave identically in training and evaluation.
- `train_countdown_grpo.py` and `eval_countdown.py` both import it. They do not import each other.
- `null_model.py` imports `build_eval_set` and `fingerprint` from `eval_countdown.py`, so it enumerates exactly the evaluation puzzles.
- `compare_evals.py` reads only the JSON files, never a model. It can run on a laptop without a GPU.

## File by file

### `countdown_grpo_starter.py`

Task definition, with no dependency on a model or on TRL.

| Name | Role |
|---|---|
| `generate_dataset(n_puzzles, n_numbers, max_val, target_range, seed)` | Generates solvable puzzles (a target reachable from the numbers). Deterministic given the seed. |
| `PROMPT_TEMPLATE`, `build_prompt(puzzle)` | French prompt with two solved examples. Applied through the chat template. |
| `reward_format` | 1.0 if the output has the think and answer tags. |
| `reward_valid` | 1.0 if the answer uses only allowed characters, the puzzle's numbers at most once each, at least two numbers, and has a finite value. |
| `reward_correctness` | 1.0 if the expression evaluates to the target. |
| `REWARD_WEIGHTS`, `total_reward` | Weights (0.1, 0.1, 1.0) and their weighted sum. |
| `completion_text`, `extract_answer` | Helpers to read a completion in string or chat format. |

Running the file directly (`python countdown_grpo_starter.py`) prints a few prompts and checks the rewards on hand-made completions.

### `train_countdown_grpo.py`

| Name | Role |
|---|---|
| `build_hf_dataset(...)` | Builds the Hugging Face dataset in conversational format (2,000 puzzles, seed 0). |
| `format_reward`, `valid_reward`, `correct_reward` | Wrappers with the signature TRL expects. Three separate functions so TRL logs each term (`rewards/<name>/mean`). Exceptions are caught and give 0.0. |
| `REWARD_FUNCS` | The three wrappers, in the order matching `REWARD_WEIGHTS`. |
| `peft_config` | LoRA: r=16, alpha=32, dropout 0.05, all linear layers. |
| `make_config(**overrides)` | `GRPOConfig` with every hyperparameter explicit. Override with keywords, for example `make_config(max_steps=150)`. |
| `main()` | Trains with the defaults and saves the adapter. |

Run `python train_countdown_grpo.py` for the default 60-step run, or use the notebook to go to 150 steps.

### `eval_countdown.py`

| Name | Role |
|---|---|
| `MODES` | `short` (50 puzzles), `mid` (300, greedy), `long` (300 greedy, 100 x 4 sampled, 100 hard), `samp` (300 greedy and 200 x 4 sampled), `hard` (100 hard, greedy and sampled). |
| `build_eval_set`, `fingerprint` | Fixed evaluation puzzles, excluded from the training puzzles, with a hash used to refuse comparisons across different sets. |
| `load_model`, `generate_batch` | Loads the base model plus an optional LoRA adapter, then batched generation (greedy or sampled). |
| `stage` | Classifies an output: broken format, forbidden characters, wrong numbers, wrong value, correct. |
| `summarize` | Accuracy, Wilson interval, bootstrap over puzzles, pass@k, stage counts, per-puzzle detail. |
| `run_eval`, `print_report`, `save` | Orchestration, console report, JSON output and `history.jsonl`. |

Main options: `--mode`, `--adapter` (path to a LoRA checkpoint, omit for the baseline), `--tag` (label in the file name), `--temperature`, `--seed`, `--skip-greedy`, `--out-dir`, `--max-new-tokens`, `--batch-size`.

### `compare_evals.py`

Takes two JSON files (baseline first) and one section: `greedy`, `sampled`, `hard`, `hard_sampled`.

- Greedy and hard: paired table, exact McNemar test, bootstrap interval on the difference, failure stage transitions.
- Sampled and hard_sampled: per-puzzle mean accuracy, bootstrap interval, pass@k, exact sign test.
- It refuses to compare results with different set fingerprints, misaligned puzzles, or different sampling temperatures.

### `null_model.py`

Exact enumeration of all expressions for each evaluation puzzle, giving the accuracy of a uniformly random valid expression (the reference for accuracy given validity). Option: `--n` (number of puzzles).

## Invariants to keep when changing the code

- **Train and eval puzzles must stay aligned.** `build_hf_dataset` defaults must match `TRAIN_SEED`, `TRAIN_N_PUZZLES` and `TRAIN_KWARGS` in `eval_countdown.py`. This is what excludes the training puzzles from the evaluation set. Change one, change the other.
- **Changing the evaluation set changes the fingerprint**, and old results then cannot be compared with new ones (by design).
- **Same rewards everywhere.** Rewards live only in `countdown_grpo_starter.py`. The evaluation stage logic (`stage`) is slightly broader than `reward_valid` (a one-number answer reaches "wrong value"), see the report.
- **Use the chat template.** Without it the Instruct model never emits an end-of-sequence token and every completion is truncated.
- **Compare like with like.** Do not compare a sampled run at one temperature with another at a different temperature (the script refuses).

## Evaluation output format

Each run writes `results/<timestamp>_<mode>_<tag>.json` with: `timestamp`, `mode`, `tag`, `adapter`, `base_model`, `temperature`, `seed`, `eval_set_hash`, `hard_set_hash` (if used), and the sections `greedy`, `sampled`, `hard`, `hard_sampled` present for that mode. Each section holds the accuracy, confidence intervals, stage counts and a `per_puzzle` list (numbers, target, correctness, stage, and the completion text for greedy runs). `history.jsonl` gets one summary line per run, which is handy to scan all runs.

## Reproduction

Open `countdown_grpo.ipynb` in Colab (GPU runtime) and run the cells in order. It installs the dependencies, applies the T4 workaround when needed, trains 60 then 150 steps (resumable from Drive checkpoints), runs the baseline and trained-model evaluations, the paired comparisons and the null model.

Before running, set the Drive paths `OUT` (checkpoints, adapter) and `RES` (evaluation JSON files) in the setup cells, and the repository URL or upload the five `.py` files. Record the library versions printed by the notebook next to the results.

## Notes

- Steps 101 to 150 ran on a T4 with TRL 1.15.0, where a bf16 to fp32 matrix multiplication path in TRL's chunked log-probability kernel is unsupported. The workaround was `import trl.kernels.chunked_logprob as k; k._MM_OUT_DTYPE = False` before creating the trainer. See the report for the consequences.
- Everything runs on a single GPU (Colab T4 was enough for training and evaluation).
