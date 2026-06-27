"""
Rafraîchit data/results.csv avec la dernière version du jeu de données
martj42/international_results (licence CC0), qui inclut les résultats des
matchs internationaux les plus récents.

Usage : python src/update_data.py
"""

import os

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_PATH = os.path.join(ROOT, "data", "results.csv")


def main():
    print(f"Téléchargement depuis {SOURCE_URL} ...")
    df = pd.read_csv(SOURCE_URL)
    df.to_csv(DEST_PATH, index=False)
    print(f"{len(df)} matchs enregistrés dans {DEST_PATH}")
    print("  Pense à relancer `python src/train_model.py` pour mettre le modèle à jour.")


if __name__ == "__main__":
    main()
