"""
Modèle nul pour Countdown : que vaut une expression valide tirée AU HASARD ?

Pourquoi : la précision conditionnelle P(correct | valide) du modèle (environ 28 % en
échantillonné) ne veut rien dire seule. Si une expression valide aléatoire atteint déjà
la cible dans une proportion comparable, alors le modèle ne cherche pas, il produit des
expressions valides qui touchent la cible par chance. Ce script calcule cette référence.

Le calcul est EXACT (énumération complète), pas une simulation : pour chaque puzzle, on
énumère toutes les expressions possibles et on compte celles qui valent la cible.

Une expression est : un sous-ensemble ordonné des nombres du puzzle (au moins 2 nombres,
chacun au plus une fois), un parenthésage (arbre binaire) et un opérateur parmi + - * /
à chaque noeud. Les divisions par zéro sont exclues (le reward les juge invalides).

Deux modèles nuls :
  - "tous les nombres" : l'expression utilise les n nombres du puzzle ;
  - "taille libre"     : on tire d'abord le nombre de nombres utilisés k entre 2 et n
                         (uniformément), puis une expression uniforme de cette taille.

La précision "solvable" donne en plus la proportion de puzzles qui ont AU MOINS une
solution (utile pour borner la précision maximale possible).

Usage :
  python null_model.py                 # jeu d'évaluation standard (300 puzzles)
  python null_model.py --n 200         # préfixe de 200 puzzles (celui du mode samp)
"""

import argparse
import itertools
import random
from fractions import Fraction

from eval_countdown import build_eval_set, fingerprint

OPS = "+-*/"


def all_values(seq):
    """Valeurs de toutes les expressions (arbres binaires x opérateurs) sur la séquence
    ordonnée seq. None = division par zéro quelque part."""
    if len(seq) == 1:
        return [Fraction(seq[0])]
    out = []
    for i in range(1, len(seq)):
        left, right = all_values(seq[:i]), all_values(seq[i:])
        for a in left:
            for b in right:
                if a is None or b is None:
                    out.extend([None] * 4)
                    continue
                out.append(a + b)
                out.append(a - b)
                out.append(a * b)
                out.append(a / b if b != 0 else None)
    return out


def stats_for_size(numbers, target, k):
    """(expressions valides, expressions justes) pour les expressions utilisant k nombres."""
    valid = hits = 0
    for idx in itertools.combinations(range(len(numbers)), k):
        for perm in itertools.permutations(idx):
            for v in all_values([numbers[i] for i in perm]):
                if v is None:
                    continue
                valid += 1
                hits += v == target
    return valid, hits


def puzzle_nulls(numbers, target):
    n = len(numbers)
    per_size = {k: stats_for_size(numbers, target, k) for k in range(2, n + 1)}
    p_all = per_size[n][1] / per_size[n][0]
    p_mixed = sum(h / v for v, h in per_size.values()) / len(per_size)
    solvable = any(h > 0 for _, h in per_size.values())
    return p_all, p_mixed, solvable


def bootstrap_ci(values, n_boot=5000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser(description="Modèle nul exact pour Countdown")
    ap.add_argument("--n", type=int, default=300, help="nombre de puzzles du jeu d'évaluation")
    args = ap.parse_args()

    puzzles = build_eval_set(300)[:args.n]
    print(f"Jeu d'évaluation : {len(puzzles)} puzzles (empreinte du jeu complet de 300 : "
          f"{fingerprint(build_eval_set(300))})\n")

    rows = [puzzle_nulls(p["numbers"], p["target"]) for p in puzzles]
    p_all = [r[0] for r in rows]
    p_mixed = [r[1] for r in rows]
    solv = [float(r[2]) for r in rows]

    def show(name, vals):
        lo, hi = bootstrap_ci(vals)
        print(f"{name:34s} {sum(vals) / len(vals):6.1%}   IC95 bootstrap [{lo:.1%} ; {hi:.1%}]")

    print("Précision d'une expression valide tirée au hasard (= P(correct | valide) du nul) :")
    show("  tous les nombres utilisés", p_all)
    show("  taille libre (2 à n nombres)", p_mixed)
    print()
    show("Puzzles solvables (>= 1 solution)", solv)

    print("\nÀ comparer, pour le modèle (mesures de l'évaluation) :")
    print("  échantillonné T=0,7 : baseline 28,6 %, v3 28,4 %")
    print("  glouton             : baseline 20,8 %, v3 16,5 %")
    print("\nLecture : si le modèle n'est pas nettement au-dessus du nul, rien ne montre qu'il")
    print("cherche. S'il est en dessous, ses expressions valides sont pires que le hasard.")


if __name__ == "__main__":
    main()
