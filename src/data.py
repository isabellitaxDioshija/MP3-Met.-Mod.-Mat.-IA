"""Carga y preparación de datos: EA Sports FC 24 (male_players.csv)."""
import pandas as pd

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
    "GK": "POR", "CB": "DEF", "FB": "DEF", "WB": "DEF",
    "CDM": "MED", "CM": "MED", "CAM": "MED", "W": "DEL", "ST": "DEL",
}

def load(path="data/male_players.csv", version=24):
    cols = ["player_id", "short_name", "player_positions", "overall", "fifa_version"] + FEATURES
    df = pd.read_csv(path, usecols=cols)
    df = df[df.fifa_version == version].drop(columns="fifa_version").copy()
    df["pos_raw"] = df.player_positions.str.split(",").str[0].str.strip()
    df["y_fine"] = df.pos_raw.map(FINE_MAP)
    df["y_macro"] = df.y_fine.map(MACRO_MAP)
    assert df[FEATURES].isna().sum().sum() == 0, "hay NaN en las features"
    return df.reset_index(drop=True)
