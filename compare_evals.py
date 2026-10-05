"""
Comparaison appariée de deux résultats d'évaluation produits par eval_countdown.py.

Usage :
  python compare_evals.py A.json B.json            # compare B à A sur les puzzles déterministes
  python compare_evals.py A.json B.json --section hard

Pourquoi un test apparié : A et B sont évalués sur les MÊMES puzzles. Ce qui compte, ce sont
les puzzles sur lesquels les deux modèles divergent. Le test de McNemar exact ne regarde que
ceux-là, donc il détecte une petite différence bien mieux qu'une comparaison d'intervalles
de confiance calculés séparément.

Si les résultats n'ont pas de détail par puzzle (anciens fichiers), on retombe sur un test de
deux proportions non apparié, moins puissant, et on le signale.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter


def mcnemar_exact(b, c):
    """p-valeur bilatérale exacte. b : puzzles réussis par A seul, c : par B seul.
    Sous l'hypothèse nulle, chaque puzzle discordant va d'un côté ou de l'autre à pile ou face."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def two_proportion_z(x1, n1, x2, n2):
    """Test de deux proportions non apparié (approximation normale, bilatéral)."""
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x2 / n2 - x1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def bootstrap_diff_ci(diffs, n_boot=10000, seed=0):
    """IC95 de la différence moyenne B moins A, en rééchantillonnant les puzzles."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label(r):
    return f"{r.get('tag') or '-'} ({r.get('adapter') or 'base'})"


def main():
    ap = argparse.ArgumentParser(description="Comparaison appariée de deux évaluations")
    ap.add_argument("a", help="JSON de référence (A), par exemple la baseline")
    ap.add_argument("b", help="JSON à comparer (B), par exemple le modèle entraîné")
    ap.add_argument("--section", choices=["greedy", "hard"], default="greedy")
    args = ap.parse_args()

    ra, rb = load(args.a), load(args.b)
    hash_key = "eval_set_hash" if args.section == "greedy" else "hard_set_hash"
    if args.section not in ra or args.section not in rb:
        sys.exit(f"Section '{args.section}' absente de l'un des fichiers (mode long requis pour 'hard').")
    if ra.get(hash_key) != rb.get(hash_key):
        sys.exit(f"Comparaison invalide : jeux d'évaluation différents "
                 f"({ra.get(hash_key)} contre {rb.get(hash_key)}).")

    sa, sb = ra[args.section], rb[args.section]
    print(f"A : {label(ra)}\nB : {label(rb)}")
    print(f"Jeu d'évaluation : {args.section}, empreinte {ra[hash_key]}, {sa['n_puzzles']} puzzles\n")

    paired = "per_puzzle" in sa and "per_puzzle" in sb
    n = sa["n_puzzles"]

    if not paired:
        xa, xb = round(sa["accuracy"] * n), round(sb["accuracy"] * n)
        p = two_proportion_z(xa, n, xb, n)
        print("ATTENTION : pas de détail par puzzle dans l'un des fichiers, comparaison NON appariée")
        print("(moins puissante). Relance les évaluations avec la version actuelle d'eval_countdown.py.\n")
        print(f"Précision A : {sa['accuracy']:.1%} ({xa}/{n})")
        print(f"Précision B : {sb['accuracy']:.1%} ({xb}/{n})")
        print(f"Test de deux proportions : p = {p:.3f}")
        return

    pa, pb = sa["per_puzzle"], sb["per_puzzle"]
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x["numbers"] != y["numbers"] or x["target"] != y["target"]:
            sys.exit(f"Puzzles non alignés à l'indice {i} : comparaison impossible.")

    ca = [row["correct"][0] for row in pa]
    cb = [row["correct"][0] for row in pb]
    both = sum(x and y for x, y in zip(ca, cb))
    only_a = sum(x and not y for x, y in zip(ca, cb))
    only_b = sum(y and not x for x, y in zip(ca, cb))
    neither = n - both - only_a - only_b

    xa, xb = both + only_a, both + only_b
    diffs = [int(y) - int(x) for x, y in zip(ca, cb)]
    lo, hi = bootstrap_diff_ci(diffs)
    p = mcnemar_exact(only_a, only_b)

    print(f"Précision A : {xa / n:.1%} ({xa}/{n})")
    print(f"Précision B : {xb / n:.1%} ({xb}/{n})")
    print(f"Différence B moins A : {(xb - xa) / n:+.1%}   IC95 bootstrap [{lo:+.1%} ; {hi:+.1%}]\n")

    print("Tableau apparié (nombre de puzzles) :")
    print(f"  réussis par les deux     : {both}")
    print(f"  réussis par A seul       : {only_a}")
    print(f"  réussis par B seul       : {only_b}")
    print(f"  réussis par aucun        : {neither}\n")

    print(f"Test de McNemar exact : p = {p:.4f}  ({only_a} + {only_b} = {only_a + only_b} puzzles discordants)")
    if only_a + only_b < 10:
        print("  Très peu de puzzles discordants : le test a peu de puissance, la conclusion est fragile.")
    if p < 0.05:
        print(f"  Différence significative à 5 % en faveur de {'B' if only_b > only_a else 'A'}.")
    else:
        print("  Pas de différence significative à 5 %. Cela ne prouve pas l'absence d'effet : "
              "ça veut dire que ces données ne le montrent pas.")

    trans = Counter((x["stage"][0], y["stage"][0]) for x, y in zip(pa, pb))
    print("\nTransitions d'étape d'échec, A vers B (les plus fréquentes) :")
    for (sa_, sb_), c in trans.most_common(10):
        mark = "" if sa_ == sb_ else "  *"
        print(f"  {sa_:26} -> {sb_:26} {c:4d}{mark}")
    print("  (* = le puzzle a changé d'étape)")


if __name__ == "__main__":
    main()
