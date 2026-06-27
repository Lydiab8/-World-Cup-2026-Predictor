"""
Entraîne le modèle prédictif et calcule les ratings Elo de toutes les
équipes à partir de l'historique complet des matchs.

Génère dans models/ :
    - goal_model.pkl     : régression de Poisson (buts attendus vs écart Elo)
    - elo_ratings.csv     : rating Elo final de chaque équipe (après le
                            dernier match connu dans le fichier de données)
    - metrics.json        : indicateurs de performance sur un jeu de test

Important : le modèle est entraîné de façon SYMÉTRIQUE. Chaque match
historique génère deux lignes d'entraînement :
    (elo_team_A - elo_team_B) -> buts marqués par l'équipe A
    (elo_team_B - elo_team_A) -> buts marqués par l'équipe B
Aucune colonne "domicile"/"extérieur" n'est utilisée comme feature : seul
l'écart de force relatif (Elo) entre les deux équipes compte. C'est ce qui
permet d'appliquer le modèle tel quel à des matchs de Coupe du Monde joués
sur terrain neutre.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from elo import compute_elo_history  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "results.csv")
MODELS_DIR = os.path.join(ROOT, "models")
TRAIN_SINCE = "1990-01-01"
MAX_GOALS = 7
ELO_SCALE = 400.0  # échelle logistique standard de l'Elo (évite les soucis
# numériques de la régression de Poisson sur des écarts de centaines de points)


def _poisson_pmf(k, lam):
    import math

    lam = max(lam, 1e-6)
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _score_matrix(lambda1, lambda2, max_goals=MAX_GOALS):
    p1 = np.array([_poisson_pmf(k, lambda1) for k in range(max_goals + 1)])
    p2 = np.array([_poisson_pmf(k, lambda2) for k in range(max_goals + 1)])
    m = np.outer(p1, p2)
    return m / m.sum()


def _outcome_probs(lambda1, lambda2):
    m = _score_matrix(lambda1, lambda2)
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def build_training_table(history: pd.DataFrame) -> pd.DataFrame:
    played = history.dropna(subset=["home_score", "away_score"]).copy()
    played = played[played["date"] >= TRAIN_SINCE].reset_index(drop=True)
    return played


def main():
    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    print("Calcul de l'historique Elo (sans avantage du terrain)...")
    history, final_ratings = compute_elo_history(df)

    played = build_training_table(history)
    print(f"{len(played)} matchs joués depuis {TRAIN_SINCE} utilisés pour l'entraînement.")

    # Split chronologique : 85% entraînement / 15% test (les plus récents)
    split_idx = int(len(played) * 0.85)
    train_df = played.iloc[:split_idx]
    test_df = played.iloc[split_idx:]

    def symmetric_xy(d):
        diff_a = (d["elo_home"] - d["elo_away"]) / ELO_SCALE
        diff_b = (d["elo_away"] - d["elo_home"]) / ELO_SCALE
        X = pd.concat([diff_a, diff_b]).values.reshape(-1, 1)
        y = pd.concat([d["home_score"], d["away_score"]]).values
        return X, y

    X_train, y_train = symmetric_xy(train_df)

    print("Entraînement de la régression de Poisson (buts attendus)...")
    model = PoissonRegressor(alpha=1e-3, max_iter=1000)
    model.fit(X_train, y_train)

    # ---- Évaluation sur le jeu de test (chronologique, donc réaliste) ----
    print("Évaluation...")
    correct = 0
    abs_err_goals = []
    exact_score_hits = 0
    for row in test_df.itertuples(index=False):
        diff = (row.elo_home - row.elo_away) / ELO_SCALE
        lam1 = max(float(model.predict([[diff]])[0]), 0.05)
        lam2 = max(float(model.predict([[-diff]])[0]), 0.05)

        p_win, p_draw, p_loss = _outcome_probs(lam1, lam2)
        if p_win >= p_draw and p_win >= p_loss:
            pred_outcome = "home"
        elif p_loss >= p_draw:
            pred_outcome = "away"
        else:
            pred_outcome = "draw"

        if row.home_score > row.away_score:
            actual_outcome = "home"
        elif row.home_score < row.away_score:
            actual_outcome = "away"
        else:
            actual_outcome = "draw"

        correct += int(pred_outcome == actual_outcome)
        abs_err_goals.append(abs(lam1 - row.home_score))
        abs_err_goals.append(abs(lam2 - row.away_score))

        m = _score_matrix(lam1, lam2)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        if i == row.home_score and j == row.away_score:
            exact_score_hits += 1

    n_test = len(test_df)
    accuracy = correct / n_test if n_test else float("nan")
    exact_score_rate = exact_score_hits / n_test if n_test else float("nan")
    mae_goals = float(np.mean(abs_err_goals)) if abs_err_goals else float("nan")

    # Baseline naïve : toujours prédire "victoire à domicile" (pour comparaison,
    # même si l'app n'utilise plus de notion domicile/extérieur)
    naive_baseline = float((test_df["home_score"] > test_df["away_score"]).mean())

    metrics = {
        "n_train_matches": int(len(train_df)),
        "n_test_matches": int(n_test),
        "winner_accuracy": round(accuracy, 4),
        "naive_home_baseline": round(naive_baseline, 4),
        "exact_score_accuracy": round(exact_score_rate, 4),
        "mean_abs_error_goals": round(mae_goals, 4),
        "trained_since": TRAIN_SINCE,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "goal_model.pkl"))

    elo_df = pd.DataFrame(
        {"team": list(final_ratings.keys()), "elo": list(final_ratings.values())}
    ).sort_values("elo", ascending=False)
    elo_df.to_csv(os.path.join(MODELS_DIR, "elo_ratings.csv"), index=False)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nModèle et ratings sauvegardés dans {MODELS_DIR}/")


if __name__ == "__main__":
    main()
