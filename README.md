#  World Cup 2026 Predictor

Application de prédiction de résultats **et de scores exacts** pour les matchs de la **Coupe du Monde FIFA 2026**, basée sur un système de rating **Elo** calculé sur **49 000+ matchs internationaux depuis 1872**, combiné à une **régression de Poisson** (scikit-learn). Le tournoi est en cours au moment de la création de ce projet — l'application intègre les résultats déjà joués et prédit les matchs à venir en temps quasi réel.

>  Projet de portfolio réalisé pour mettre en pratique des compétences de Data Science / IA (collecte & prétraitement de données, feature engineering, modélisation probabiliste, interface Streamlit) sur un sujet grand public.

##  Fonctionnalités

- ** Prédicteur de match** — choisis deux équipes (parmi les 300+ sélections nationales de l'historique) et obtiens : la probabilité de victoire / nul / défaite, **le score exact le plus probable**, les buts attendus (xG) de chaque équipe, et le détail des 5 scénarios de score les plus probables
- ** Prochains matchs de la CDM 2026** — prédictions automatiques (résultat + score) pour tous les matchs programmés et pas encore joués, calculées à partir du calendrier réel du tournoi
- ** Classement Elo** — classement à jour des 48 équipes qualifiées, par groupe (A à L)

Aucune notion de **domicile/extérieur** n'intervient nulle part dans l'application : la Coupe du Monde se joue presque exclusivement sur sites neutres, et le modèle a été conçu spécifiquement pour refléter ça (voir ci-dessous).

## Comment ça marche

1. **Données** — historique complet des matchs internationaux ([martj42/international_results](https://github.com/martj42/international_results), licence CC0), mis à jour en continu avec les résultats de la Coupe du Monde 2026
2. **Rating Elo — sans avantage du terrain**  chaque équipe démarre à 1500 points ; son rating évolue après chaque match selon le résultat, l'écart de buts et l'importance de la compétition (un match de Coupe du Monde pèse plus qu'un match amical). **Aucun bonus n'est appliqué à l'équipe qui reçoit** : le calcul est purement symétrique, pour éviter qu'un avantage du terrain appris sur 150 ans de matchs amicaux ne biaise les prédictions d'un tournoi joué sur sites neutres. Le calcul est fait match par match dans l'ordre chronologique, sans aucune fuite d'information vers le futur
3. **Modèle prédictif — régression de Poisson** — un modèle de buts attendus  est entraîné sur les matchs depuis 1990, à partir du seul écart de rating Elo entre les deux équipes. Chaque match historique est dédoublé symétriquement à l'entraînement (équipe A vs B *et* B vs A) pour garantir que le modèle ne retienne aucun biais lié à l'ordre des équipes. À partir des deux buts attendus (un par équipe), on construit la matrice de probabilité de chaque score exact (loi de Poisson) : le score le plus probable de cette matrice devient la prédiction de score, et les probabilités victoire/nul/défaite s'obtiennent en sommant les cases correspondantes
4. **Interface** — une application Streamlit interactive expose ces prédictions

##  Stack technique

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Streamlit`

##  Installation & lancement en local

```bash
git clone https://github.com/<ton-pseudo-github>/worldcup-2026-predictor.git
cd worldcup-2026-predictor
pip install -r requirements.txt

# (optionnel) récupérer les tout derniers résultats avant d'entraîner
python src/update_data.py

# entraîner le modèle (génère models/goal_model.pkl, models/elo_ratings.csv, models/metrics.json)
python src/train_model.py

# lancer l'application
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.



##  Structure du projet

```
worldcup-2026-predictor/
├── app.py                  # Application Streamlit
├── requirements.txt
├── data/
│   └── results.csv          # Historique des matchs internationaux (1872 -> aujourd'hui)
├── src/
│   ├── elo.py               # Calcul des ratings Elo (sans avantage du terrain)
│   ├── train_model.py       # Entraînement du modèle (régression de Poisson)
│   ├── predict.py           # Prédiction : probabilités + score exact
│   ├── teams_2026.py        # Les 12 groupes de la CDM 2026
│   └── update_data.py       # Rafraîchissement des données
└── models/                  # Artefacts générés (goal_model.pkl, elo_ratings.csv, metrics.json)
```

##  Performance du modèle

Sur un jeu de test chronologique (15 % des matchs les plus récents depuis 1990, jamais vus à l'entraînement) :

| Métrique | Valeur |
|---|---|
| Précision du résultat (victoire/nul/défaite) | ~59 % |
| Précision du score exact | ~13 % |
| Erreur absolue moyenne sur les buts | ~0.96 but |

À titre de comparaison, prédire systématiquement la victoire de l'équipe à domicile (stratégie naïve) donne environ 48 % de précision sur ce jeu de données : le modèle apporte donc un vrai gain, **même sans utiliser aucune information de terrain**. Deviner le score exact d'un match de football reste un exercice statistiquement très difficile — ~13 % est cohérent avec ce que rapportent les modèles de prédiction sportive grand public. Comme souvent en prédiction sportive, **le match nul reste la classe la plus difficile à anticiper** — c'est une limite connue de ce type d'approche, pas un bug.

##  Limites & disclaimer

Ce modèle se base uniquement sur la force relative des équipes (Elo). Il ne tient pas compte des blessures, de la composition d'équipe, de la forme du moment ou de la tactique, et le score exact prédit n'est que le scénario le plus probable parmi des dizaines d'issues possibles. **Ce projet est un exercice de portfolio, pas un outil d'aide à la décision pour des paris sportifs.**

##  Pistes d'amélioration

- Remplacer/compléter la régression de Poisson par un modèle de **Dixon-Coles** (corrélation entre les buts des deux équipes) pour affiner encore les scores prédits
- Ajouter un **LSTM** sur les séquences de résultats récents de chaque équipe, pour mieux capturer la dynamique de forme
- Ajouter un mini **agent de Reinforcement Learning** (Gymnasium + Stable-Baselines3) simulant des tirs au but, pour explorer la prise de décision optimale dans un contexte simplifié
- Intégrer des données complémentaires : classement FIFA officiel, valeur marchande des effectifs, données de forme 

##  Licence

MIT — voir [LICENSE](LICENSE). Données sous licence CC0 ([source](https://github.com/martj42/international_results)).
