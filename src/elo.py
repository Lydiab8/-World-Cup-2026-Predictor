"""
Calcul des ratings Elo des équipes nationales.

Important : ce système est volontairement *neutre*. Contrairement à de
nombreuses implémentations d'Elo football qui ajoutent un bonus de ~100
points à l'équipe qui reçoit, on ne fait ici AUCUNE distinction
domicile/extérieur, ni dans le calcul du rating, ni dans les features
utilisées par le modèle prédictif. C'est un choix délibéré : à la Coupe du
Monde, l'immense majorité des matchs se jouent sur terrain neutre, et on ne
veut pas qu'un avantage du terrain appris sur 150 ans de matchs amicaux
(souvent très déséquilibrés) vienne biaiser les prédictions du tournoi.

Le rating de chaque équipe évolue uniquement en fonction de :
- l'écart de rating avec l'adversaire (résultat attendu vs résultat réel)
- l'écart de buts du match (une victoire 4-0 pèse plus qu'un 1-0)
- l'importance de la compétition (un match de Coupe du Monde pèse plus
  qu'un match amical)
"""

import pandas as pd

INITIAL_ELO = 1500.0

# Poids de base (K-factor) par type de compétition.
# Inspiré de l'échelle utilisée par eloratings.net / World Football Elo Ratings.
_TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 35,
    "Confederations Cup": 40,
    "Copa América": 50,
    "UEFA Euro": 55,
    "UEFA Euro qualification": 35,
    "UEFA Nations League": 35,
    "African Cup of Nations": 45,
    "African Cup of Nations qualification": 30,
    "AFC Asian Cup": 45,
    "AFC Asian Cup qualification": 30,
    "CONCACAF Gold Cup": 40,
    "Friendly": 20,
}
_DEFAULT_K = 25


def _k_factor(tournament: str) -> float:
    if tournament in _TOURNAMENT_K:
        return _TOURNAMENT_K[tournament]
    # recherche approximative pour les variantes de libellé non listées
    t = str(tournament).lower()
    if "world cup qualification" in t:
        return _TOURNAMENT_K["FIFA World Cup qualification"]
    if "world cup" in t:
        return _TOURNAMENT_K["FIFA World Cup"]
    if "friendly" in t:
        return _TOURNAMENT_K["Friendly"]
    if "qualif" in t:
        return 30
    return _DEFAULT_K


def _goal_diff_multiplier(goal_diff: float) -> float:
    """Plus l'écart de buts est grand, plus le match pèse dans le calcul."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def compute_elo_history(df: pd.DataFrame):
    """
    Parcourt l'historique des matchs en ordre chronologique et calcule,
    pour CHAQUE match, le rating Elo de chaque équipe juste AVANT le match
    (donc sans aucune fuite d'information vers le futur), puis met à jour
    les ratings si le score est connu.

    Retourne :
        history : DataFrame avec une ligne par match jamais joué ou joué,
                   colonnes [date, home_team, away_team, elo_home, elo_away,
                             home_score, away_score, tournament]
                   (elo_home / elo_away = rating juste avant le match,
                   "home_team"/"away_team" reflète uniquement l'ordre dans
                   le fichier source — aucun avantage n'est appliqué)
        final_ratings : dict {équipe: rating final}
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}
    records = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home = ratings.get(home, INITIAL_ELO)
        r_away = ratings.get(away, INITIAL_ELO)

        records.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "elo_home": r_home,
                "elo_away": r_away,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "tournament": row.tournament,
            }
        )

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue  # match pas encore joué : pas de mise à jour de rating

        # Formule Elo standard, SANS bonus de terrain
        expected_home = 1.0 / (1.0 + 10 ** (-(r_home - r_away) / 400.0))

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score == row.away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = _k_factor(row.tournament) * _goal_diff_multiplier(
            row.home_score - row.away_score
        )
        delta = k * (actual_home - expected_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    history = pd.DataFrame.from_records(records)
    return history, ratings


def load_upcoming_fixtures(data_dir: str) -> pd.DataFrame:
    """Charge les matchs sans score (= pas encore joués) depuis data/results.csv."""
    df = pd.read_csv(f"{data_dir}/results.csv", parse_dates=["date"])
    upcoming = df[df["home_score"].isna() | df["away_score"].isna()].copy()
    return upcoming.sort_values("date").reset_index(drop=True)
