"""Carga y preparación de datos: EA Sports FC 24."""
import os
import pandas as pd

# 34 atributos base + 2 físicos. Se excluyen a propósito:
#  - ls, st, ..., gk (rating por posición): FUGA de la etiqueta
#  - club_position / nation_position / player_positions: son la etiqueta
#  - pace, shooting, ... (6 compuestos): derivados de los base y NaN para porteros
FEATURES = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina", "power_strength",
    "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes",
    "height_cm", "weight_kg",
]

# Izquierda/derecha no es identificable desde atributos (son simétricos) -> se fusionan.
FINE_MAP = {
    "GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "LWB": "WB", "RWB": "WB",
    "CDM": "CDM", "CM": "CM", "CAM": "CAM",
    "LM": "W", "RM": "W", "LW": "W", "RW": "W", "ST": "ST", "CF": "ST",
}
MACRO_MAP = {
    "GK": "GK", "CB": "DEF", "FB": "DEF", "WB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "W": "FWD", "ST": "FWD",
}

RAW = "data/male_players.csv"       # CSV original de Kaggle (opcional, 96 MB)
LIGHT = "data/players_fc24.csv"     # subconjunto FC 24 incluido en el repo

def load(version=24):
    if os.path.exists(RAW):
        cols = ["player_id", "short_name", "player_positions", "overall", "fifa_version"] + FEATURES
        df = pd.read_csv(RAW, usecols=cols)
        df = df[df.fifa_version == version].drop(columns="fifa_version").copy()
    else:
        df = pd.read_csv(LIGHT)
    df["pos_raw"] = df.player_positions.str.split(",").str[0].str.strip()
    df["y_fine"] = df.pos_raw.map(FINE_MAP)
    df["y_macro"] = df.y_fine.map(MACRO_MAP)
    df["y_fine"] = df["y_fine"].astype(object)
    df["y_macro"] = df["y_macro"].astype(object)
    assert df[FEATURES].isna().sum().sum() == 0, "hay NaN en las features"
    return df.reset_index(drop=True)