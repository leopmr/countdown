"""
Point de départ du projet Countdown x GRPO.

Contenu :
1. Génération de puzzles Countdown (nombres + cible)
2. Template de prompt avec balises <think>/<answer>
3. Reward function de format (respecte la structure demandée)
4. Reward function de correction (l'expression évalue-t-elle à la cible,
   en utilisant uniquement les nombres fournis, chacun au plus une fois)

Ce fichier ne fait pas encore l'entraînement (ça viendra avec TRL
GRPOTrainer), c'est la brique de données + reward à valider isolément
avant de brancher un modèle, comme prévu en semaine 1 jours 1-4.
"""

import random
"""
Point de départ du projet Countdown x GRPO.

Contenu :
1. Génération de puzzles Countdown (nombres + cible)
2. Template de prompt avec balises <think>/<answer>
3. Reward function de format (respecte la structure demandée)
4. Reward function de correction (l'expression évalue-t-elle à la cible,
   en utilisant uniquement les nombres fournis, chacun au plus une fois)

Ce fichier ne fait pas encore l'entraînement (ça viendra avec TRL
GRPOTrainer), c'est la brique de données + reward à valider isolément
avant de brancher un modèle, comme prévu en semaine 1 jours 1-4.
"""

import random
import re
import itertools
import operator


# ---------------------------------------------------------------------------
# 1. Génération de puzzles
# ---------------------------------------------------------------------------

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _random_solvable_puzzle(n_numbers=4, min_val=1, max_val=25, target_range=(10, 999)):
    """Génère un puzzle en partant d'une cible garantie atteignable :
    on tire des nombres au hasard, on les combine avec des opérations
    aléatoires pour produire la cible, puis on ne garde que le résultat
    final (le modèle devra retrouver un chemin, pas forcément celui-ci)."""
    numbers = [random.randint(min_val, max_val) for _ in range(n_numbers)]
    pool = numbers.copy()
    random.shuffle(pool)

    value = pool[0]
    for n in pool[1:]:
        op = random.choice(list(OPS.values()))
        try:
            candidate = op(value, n)
        except ZeroDivisionError:
            candidate = value + n
        # on évite les valeurs non entières ou négatives pour rester simple
        if candidate != int(candidate) or candidate < 0:
            candidate = value + n
        value = candidate

    target = int(value)
    if not (target_range[0] <= target <= target_range[1]):
        return _random_solvable_puzzle(n_numbers, min_val, max_val, target_range)

    return {"numbers": numbers, "target": target}


def generate_dataset(n_puzzles=1000, n_numbers=4, seed=0):
    random.seed(seed)
    return [_random_solvable_puzzle(n_numbers=n_numbers) for _ in range(n_puzzles)]


# ---------------------------------------------------------------------------
# 2. Template de prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Tu résous des puzzles Countdown : à partir d'une liste de nombres, \
trouve une expression arithmétique (+, -, *, /) qui donne exactement le nombre cible. \
Chaque nombre ne peut être utilisé qu'une seule fois.

Réfléchis d'abord dans des balises <think></think>, puis donne ton expression \
finale dans des balises <answer></answer>, sans rien écrire après.

Voici deux exemples résolus :

Nombres : [3, 7, 2], cible : 17
<think>
J'essaie 7 * 2 = 14, il me reste 3, 14 + 3 = 17. Ça marche.
</think>
<answer>
7 * 2 + 3
</answer>

Nombres : [10, 4, 6], cible : 20
<think>
J'essaie 10 + 4 = 14, plus 6 ça fait 20. Ça marche directement.
</think>
<answer>
10 + 4 + 6
</answer>

À ton tour.

Nombres : {numbers}, cible : {target}
"""


def build_prompt(puzzle):
    return PROMPT_TEMPLATE.format(numbers=puzzle["numbers"], target=puzzle["target"])


# ---------------------------------------------------------------------------
# 3. Reward de format
# ---------------------------------------------------------------------------

THINK_ANSWER_RE = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL
)


def reward_format(completion: str) -> float:
    """1.0 si la structure <think>...</think><answer>...</answer> apparaît
    quelque part dans la complétion, 0.0 sinon. Version assouplie : on
    cherche toutes les occurrences et on ne s'intéresse qu'à la dernière
    (la conclusion du modèle), pour rester robuste si le modèle répète
    la structure plusieurs fois dans une même génération."""
    return 1.0 if list(THINK_ANSWER_RE.finditer(completion.strip())) else 0.0


def extract_answer(completion: str):
    matches = list(THINK_ANSWER_RE.finditer(completion.strip()))
    if not matches:
        return None
    return matches[-1].group(1).strip()


# ---------------------------------------------------------------------------
# 4. Reward de correction
# ---------------------------------------------------------------------------

ALLOWED_CHARS = re.compile(r"^[0-9+\-*/(). ]+$")


def _extract_numbers_used(expression: str):
    return [int(tok) for tok in re.findall(r"\d+", expression)]


def _uses_numbers_correctly(expression: str, numbers: list[int]) -> bool:
    """Vérifie que chaque nombre utilisé dans l'expression fait bien
    partie du puzzle, et qu'aucun nombre n'est utilisé plus de fois que
    disponible dans la liste d'origine (règle du jeu Countdown)."""
    used = _extract_numbers_used(expression)
    available = numbers.copy()
    for n in used:
        if n in available:
            available.remove(n)
        else:
            return False
    return True


def reward_correctness(completion: str, numbers: list[int], target: int) -> float:
    """1.0 si l'expression extraite évalue exactement à target, en
    n'utilisant que les nombres fournis (au plus une fois chacun),
    0.0 sinon."""
    expression = extract_answer(completion)
    if expression is None:
        return 0.0
    if not ALLOWED_CHARS.match(expression):
        return 0.0
    if not _uses_numbers_correctly(expression, numbers):
        return 0.0
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # nombres/opérateurs uniquement
    except Exception:
        return 0.0
    return 1.0 if abs(value - target) < 1e-4 else 0.0


def total_reward(completion: str, numbers: list[int], target: int,
                  format_weight=0.1, correctness_weight=1.0) -> float:
    return (format_weight * reward_format(completion)
            + correctness_weight * reward_correctness(completion, numbers, target))


# ===========================================================================
# TESTS / DÉMO
# ===========================================================================

if __name__ == "__main__":
    dataset = generate_dataset(n_puzzles=5, seed=0)
    for puzzle in dataset:
        print(build_prompt(puzzle))
        print("---")

    puzzle = {"numbers": [25, 8, 3, 7], "target": 56}

    good_completion = "<think>7 * 8 = 56</think>\n<answer>7 * 8</answer>"
    bad_format = "7 * 8 = 56"
    wrong_number = "<think>x</think>\n<answer>7 * 9</answer>"  # 9 pas dans la liste
    wrong_result = "<think>x</think>\n<answer>25 + 8</answer>"  # ne donne pas 56

    print("format(good) :", reward_format(good_completion))       # 1.0
    print("format(bad)  :", reward_format(bad_format))            # 0.0
    print("correctness(good)       :", reward_correctness(good_completion, puzzle["numbers"], puzzle["target"]))  # 1.0
    print("correctness(wrong_num)  :", reward_correctness(wrong_number, puzzle["numbers"], puzzle["target"]))     # 0.0
    print("correctness(wrong_res)  :", reward_correctness(wrong_result, puzzle["numbers"], puzzle["target"]))    # 0.0
    print("total(good)             :", total_reward(good_completion, puzzle["numbers"], puzzle["target"]))       # 1.1