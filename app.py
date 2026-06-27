import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from elo import load_upcoming_fixtures
from predict import load_artifacts, predict_match
from teams_2026 import GROUPS_2026, all_teams_2026, team_group

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Identité visuelle — palette "tableau de marque", typographies condensées /
# monospace façon écran de stade, injectées une seule fois au chargement.
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg-deep: #0E1A16;
    --bg-card: #14241F;
    --bg-card-alt: #1B2E27;
    --line: rgba(255,255,255,0.08);
    --accent-gold: #F2B807;
    --accent-teal: #2BD9A6;
    --text-chalk: #F3F6F2;
    --text-muted: #8FA39A;
}

.stApp {
    background: radial-gradient(circle at 15% -10%, #173026 0%, var(--bg-deep) 55%);
    font-family: 'Inter', sans-serif;
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li {
    color: var(--text-chalk);
}
[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
}

.stApp h2, .stApp h3 {
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-size: 1.05rem !important;
    color: var(--text-chalk) !important;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.4rem;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.72rem;
    color: var(--accent-teal);
    margin-bottom: 0.3rem;
}
.hero-title {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    font-size: 2.5rem;
    margin: 0.1rem 0 0.5rem 0;
}
.hero-rule {
    height: 3px;
    width: 64px;
    background: var(--accent-gold);
    border-radius: 2px;
    margin-bottom: 0.9rem;
}
.hero-caption {
    color: var(--text-muted);
    font-size: 0.95rem;
    line-height: 1.55;
    max-width: 780px;
}

div[data-baseweb="tab-list"] {
    gap: 1.6rem;
    border-bottom: 1px solid var(--line);
}
button[data-baseweb="tab"] {
    font-family: 'Oswald', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-size: 0.85rem;
    color: var(--text-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-chalk) !important;
    border-bottom: 3px solid var(--accent-gold) !important;
}

.match-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(180deg, var(--bg-card-alt), var(--bg-card));
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.3rem 1.8rem;
    margin: 0.6rem 0 1rem 0;
}
.team { flex: 1; text-align: center; }
.team-name {
    font-family: 'Oswald', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    font-size: 1.05rem;
}
.team-elo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}
.score-display {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 2.3rem;
    color: var(--accent-gold);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0 1.5rem;
}
.score-sep { color: var(--text-muted); font-weight: 500; font-size: 1.6rem; }

.prob-bar {
    display: flex;
    height: 28px;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid var(--line);
}
.prob-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--bg-deep);
}
.prob-team1 { background: var(--accent-gold); }
.prob-draw  { background: #5B6B65; color: var(--text-chalk); }
.prob-team2 { background: var(--accent-teal); }
.prob-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.74rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
}

.badge {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 0.15rem;
    background: var(--bg-card-alt);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
}
.badge-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent-gold);
}
.badge-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
}

.callout {
    background: var(--bg-card);
    border-left: 3px solid var(--accent-teal);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.6rem;
}

.score-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.5;
    margin: 0.7rem 0 0 0;
}

div[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border-color: var(--line) !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
}
hr { border-color: var(--line) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Chargement des artefacts (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_model_and_elo():
    return load_artifacts()


@st.cache_data
def get_upcoming_fixtures():
    df = load_upcoming_fixtures("data")
    return df[df["tournament"] == "FIFA World Cup"].reset_index(drop=True)


model, elo_dict = get_model_and_elo()
upcoming = get_upcoming_fixtures()

ALL_TEAMS = sorted(elo_dict.keys())
QUALIFIED_2026 = all_teams_2026()


# ----------------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------------
st.markdown(
    """
    <p class="hero-eyebrow">Coupe du monde 2026 · Projet data science</p>
    <div class="hero-title">⚽ World Cup 2026 Predictor</div>
    <div class="hero-rule"></div>
    <p class="hero-caption">
        Prédiction de résultats <strong>et de scores exacts</strong> de matchs, basée sur un rating
        <strong>Elo</strong> calculé sur <strong>49 000+ matchs internationaux depuis 1872</strong> et un
        modèle de <strong>régression de Poisson</strong>. La Coupe du Monde 2026 est en cours — les résultats
        déjà joués sont intégrés au modèle en temps réel. Tous les matchs sont traités sur un pied d'égalité :
        <strong>aucun avantage du terrain n'est appliqué</strong>, conformément au format de la compétition,
        qui se joue presque exclusivement sur sites neutres.
    </p>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [" Prédire un match", " Prochains matchs — CDM 2026", " Classement Elo"]
)


def render_prediction(team1, team2, result, key_prefix=""):
    """Affiche le tableau de marque + barre de probabilités pour un match déjà calculé."""
    p1 = result["team1_win"] * 100
    pdraw = result["draw"] * 100
    p2 = result["team2_win"] * 100

    st.markdown(
        f"""
        <div class="match-card">
            <div class="team">
                <div class="team-name">{team1}</div>
                <div class="team-elo">ELO&nbsp;{result['elo_team1']:.0f}</div>
            </div>
            <div class="score-display">
                <span>{result['score_team1']}</span>
                <span class="score-sep">–</span>
                <span>{result['score_team2']}</span>
            </div>
            <div class="team">
                <div class="team-name">{team2}</div>
                <div class="team-elo">ELO&nbsp;{result['elo_team2']:.0f}</div>
            </div>
        </div>
        <div class="prob-bar">
            <div class="prob-seg prob-team1" style="width:{p1:.1f}%;">{p1:.0f}%</div>
            <div class="prob-seg prob-draw" style="width:{pdraw:.1f}%;">{pdraw:.0f}%</div>
            <div class="prob-seg prob-team2" style="width:{p2:.1f}%;">{p2:.0f}%</div>
        </div>
        <div class="prob-labels">
            <span>Victoire {team1}</span><span>Nul</span><span>Victoire {team2}</span>
        </div>
        <p class="score-note">
            ⓘ Score le plus probable ≠ résultat le plus probable : les victoires se répartissent sur
            plusieurs scores (1-0, 2-1, 3-1…) alors que les nuls se concentrent sur très peu de scores
            (0-0, 1-1, 2-2…). Un favori net peut donc afficher un score nul comme score le plus probable —
            ce n'est pas une contradiction du modèle.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    xg_col, score_col = st.columns([2, 1])
    with xg_col:
        st.caption(
            f"Buts attendus (xG) — {team1} : **{result['xg_team1']:.2f}** · "
            f"{team2} : **{result['xg_team2']:.2f}**"
        )
        top_df = pd.DataFrame(
            [
                {"Score": f"{i}-{j}", "Probabilité (%)": round(p * 100, 1)}
                for i, j, p in result["top_scores"]
            ]
        )
        st.dataframe(top_df, hide_index=True, use_container_width=True)
    with score_col:
        st.markdown(
            f"""
            <div class="badge">
                <span class="badge-score">{result['score_team1']} – {result['score_team2']}</span>
                <span class="badge-sub">{result['score_prob']*100:.1f}% de probabilité</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# TAB 1 — Prédire un match entre deux équipes au choix
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Choisis deux équipes")
    st.caption("Match joué sur terrain neutre — comme la quasi-totalité des matchs de la CDM 2026.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox(
            "Équipe 1",
            ALL_TEAMS,
            index=ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 0,
        )
    with col2:
        team2 = st.selectbox(
            "Équipe 2",
            ALL_TEAMS,
            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 1,
        )

    if team1 == team2:
        st.warning("Choisis deux équipes différentes.")
    else:
        result = predict_match(model, elo_dict, team1, team2)
        st.write("")
        render_prediction(team1, team2, result)

    st.markdown(
        """
        <div class="callout">
            💡 Le modèle ne connaît que la force relative des équipes (Elo). Il ne prend pas en
            compte les blessures, la forme du jour ou la tactique, et le score exact prédit est
            le résultat le plus probable parmi des dizaines de scénarios — pas une certitude.
            C'est un outil de portfolio, pas un pronostiqueur professionnel.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# TAB 2 — Prochains matchs réels de la Coupe du Monde 2026
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Prédictions pour les prochains matchs programmés")

    if upcoming.empty:
        st.info("Aucun match à venir trouvé dans les données (phase de groupes peut-être terminée).")
    else:
        rows = []
        for _, m in upcoming.iterrows():
            t1, t2 = m["home_team"], m["away_team"]
            pred = predict_match(model, elo_dict, t1, t2)
            favori = t1 if pred["team1_win"] >= pred["team2_win"] else t2
            proba_favori = max(pred["team1_win"], pred["team2_win"])
            rows.append(
                {
                    "Date": m["date"].strftime("%d/%m/%Y"),
                    "Match": f"{t1} vs {t2}",
                    "Ville": m["city"],
                    "Score prédit": f"{pred['score_team1']}-{pred['score_team2']}",
                    "Victoire Éq. 1": f"{pred['team1_win']*100:.0f} %",
                    "Nul": f"{pred['draw']*100:.0f} %",
                    "Victoire Éq. 2": f"{pred['team2_win']*100:.0f} %",
                    "Favori": f"{favori} ({proba_favori*100:.0f} %)",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"{len(rows)} matchs à venir détectés dans le calendrier de la Coupe du Monde 2026. "
            "« Éq. 1 » / « Éq. 2 » correspondent à l'ordre des équipes dans la colonne *Match*."
        )

        st.divider()
        st.subheader("Détail d'un match à venir")
        match_labels = [f"{m['home_team']} vs {m['away_team']} ({m['date'].strftime('%d/%m')})" for _, m in upcoming.iterrows()]
        choice = st.selectbox("Choisis un match", match_labels)
        chosen_idx = match_labels.index(choice)
        chosen = upcoming.iloc[chosen_idx]
        t1, t2 = chosen["home_team"], chosen["away_team"]
        result = predict_match(model, elo_dict, t1, t2)
        render_prediction(t1, t2, result, key_prefix="fixture")


# ----------------------------------------------------------------------------
# TAB 3 — Classement Elo des 48 qualifiés
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Classement Elo des 48 équipes qualifiées")

    elo_table = pd.DataFrame(
        {"Équipe": QUALIFIED_2026, "Elo": [elo_dict.get(t, 1500) for t in QUALIFIED_2026]}
    )
    elo_table["Groupe"] = elo_table["Équipe"].apply(team_group)
    elo_table = elo_table.sort_values("Elo", ascending=False).reset_index(drop=True)
    elo_table.index = elo_table.index + 1
    elo_table["Elo"] = elo_table["Elo"].round(0).astype(int)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.bar_chart(elo_table.set_index("Équipe")["Elo"].head(20), color="#2BD9A6")
    with col_b:
        st.dataframe(elo_table, use_container_width=True)

    st.divider()
    st.subheader("Vue par groupe")
    group_choice = st.selectbox("Groupe", sorted(GROUPS_2026.keys()))
    group_df = elo_table[elo_table["Groupe"] == group_choice].sort_values("Elo", ascending=False)
    st.table(group_df[["Équipe", "Elo"]])


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <p style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:var(--text-muted);">
        Données : <a href="https://github.com/martj42/international_results" style="color:var(--accent-teal);">
        martj42/international_results</a> (CC0) — Modèle : Elo (sans avantage du terrain) + régression
        de Poisson (scikit-learn) — Projet réalisé par Lydia Bouaoudia.
    </p>
    """,
    unsafe_allow_html=True,
)