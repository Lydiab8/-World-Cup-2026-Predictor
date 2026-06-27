"""
Prédiction d'un match : probabilités de résultat (victoire / nul / défaite)
ET score exact le plus probable, à partir d'un seul modèle cohérent.

Principe (modèle de Poisson, classique en analyse de matchs de football) :
1. On entraîne une régression de Poisson qui prédit le nombre de buts
   qu'une équipe devrait marquer en fonction de SON écart de rating Elo
   par rapport à l'adversaire (elo_equipe - elo_adversaire).
2. Pour un match Équipe 1 vs Équipe 2, on obtient ainsi deux "buts
   attendus" (xG) : lambda1 pour l'équipe 1, lambda2 pour l'équipe 2.
3. On construit la matrice de probabilité de chaque score exact possible
   (0-0, 1-0, 2-1, ...) en supposant que les deux équipes marquent selon
   des lois de Poisson indépendantes.
4. Le score le plus probable = la case la plus probable de cette matrice.
   Les probabilités victoire/nul/défaite = somme des cases sous/sur/dans
   la diagonale.

Aucune notion de domicile/extérieur n'intervient : le modèle est entraîné
de façon symétrique (cf. train_model.py) et ne connaît que l'écart de force
relatif entre les deux équipes — ce qui correspond à la réalité d'un match
de Coupe du Monde sur terrain neutre.
"""

import math
import os

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MAX_GOALS = 7  # au-delà de 7 buts, la probabilité est négligeable
DEFAULT_ELO = 1500.0
ELO_SCALE = 400.0  # doit être identique à celle utilisée dans train_model.py


def load_artifacts():
    """Charge le modèle de buts entraîné et le dictionnaire des ratings Elo."""
    model = joblib.load(os.path.join(MODELS_DIR, "goal_model.pkl"))
    elo_df = pd.read_csv(os.path.join(MODELS_DIR, "elo_ratings.csv"))
    elo_dict = dict(zip(elo_df["team"], elo_df["elo"]))
    return model, elo_dict


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(lam, 1e-6)
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _score_matrix(lambda1: float, lambda2: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    p1 = np.array([_poisson_pmf(k, lambda1) for k in range(max_goals + 1)])
    p2 = np.array([_poisson_pmf(k, lambda2) for k in range(max_goals + 1)])
    matrix = np.outer(p1, p2)  # matrix[i, j] = P(équipe1 marque i ET équipe2 marque j)
    matrix = matrix / matrix.sum()  # renormalisation (troncature à max_goals)
    return matrix


def predict_match(model, elo_dict, team1: str, team2: str) -> dict:
    """
    Prédit un match entre team1 et team2 sur terrain neutre.

    Retourne un dict avec :
        team1_win, draw, team2_win : probabilités (somment à 1)
        score_team1, score_team2   : score exact le plus probable
        score_prob                  : probabilité de ce score exact précis
        xg_team1, xg_team2          : buts attendus (espérance) par équipe
        elo_team1, elo_team2        : ratings Elo actuels
        top_scores                  : 5 scores les plus probables [(s1,s2,proba), ...]
    """
    elo1 = elo_dict.get(team1, DEFAULT_ELO)
    elo2 = elo_dict.get(team2, DEFAULT_ELO)
    diff = (elo1 - elo2) / ELO_SCALE

    lambda1 = float(model.predict([[diff]])[0])
    lambda2 = float(model.predict([[-diff]])[0])
    lambda1, lambda2 = max(lambda1, 0.05), max(lambda2, 0.05)

    matrix = _score_matrix(lambda1, lambda2)

    team1_win = float(np.tril(matrix, -1).sum())  # team1 marque plus que team2
    draw = float(np.trace(matrix))
    team2_win = float(np.triu(matrix, 1).sum())

    flat_idx = np.argsort(matrix, axis=None)[::-1]
    top_scores = []
    for idx in flat_idx[:5]:
        i, j = np.unravel_index(idx, matrix.shape)
        top_scores.append((int(i), int(j), float(matrix[i, j])))

    best_i, best_j, best_p = top_scores[0]

    return {
        "team1_win": team1_win,
        "draw": draw,
        "team2_win": team2_win,
        "score_team1": best_i,
        "score_team2": best_j,
        "score_prob": best_p,
        "xg_team1": lambda1,
        "xg_team2": lambda2,
        "elo_team1": elo1,
        "elo_team2": elo2,
        "top_scores": top_scores,
    }
