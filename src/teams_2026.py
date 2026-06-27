"""
Les 12 groupes (A -> L) de la Coupe du Monde FIFA 2026, tels qu'issus du
tirage officiel du 5 décembre 2025 (Mexique, Canada, États-Unis).
"""

GROUPS_2026 = {
    "A": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["Australia", "Paraguay", "Turkey", "United States"],
    "E": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["Croatia", "England", "Ghana", "Panama"],
}


def all_teams_2026():
    """Retourne la liste triée des 48 équipes qualifiées."""
    teams = []
    for group_teams in GROUPS_2026.values():
        teams.extend(group_teams)
    return sorted(teams)


def team_group(team):
    """Retourne la lettre du groupe (A-L) d'une équipe, ou '?' si inconnue."""
    for letter, teams in GROUPS_2026.items():
        if team in teams:
            return letter
    return "?"
