
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib, numpy as np, pandas as pd, streamlit as st
from data import FEATURES

st.set_page_config(page_title="¿En qué posición juega?", page_icon="⚽", layout="wide")
st.title("¿En qué posición juega? — SVM sobre EA Sports FC 24")

@st.cache_resource
def get_model(target):
    return joblib.load(ROOT / "models" / f"svm_{target}.joblib")

@st.cache_data
def get_players():
    return pd.read_csv(ROOT / "data" / "players_fc24.csv")

target = st.sidebar.radio("Granularidad", ["y_macro", "y_fine"],
        format_func=lambda t: "4 líneas (POR/DEF/MED/DEL)" if t == "y_macro" else "9 posiciones")
model = get_model(target)
players = get_players()
classes = list(model.classes_)
RENAME = {"POR": "GK", "MED": "MID", "DEL": "FWD"}   # DEF ya coincide
show = lambda c: RENAME.get(c, c)

def predict(x):
    x = np.asarray(x, float).reshape(1, -1)
    pred = model.predict(x)[0]
    s = model.decision_function(x)[0]            # puntajes OvO agregados (por clase)
    e = np.exp(s - s.max()); rel = e / e.sum()   # solo para visualizar (no es probabilidad calibrada)
    return pred, pd.Series(rel, index=classes).sort_values(ascending=False)

tab1, tab2 = st.tabs(["Jugador real", "Crear jugador"])

with tab1:
    q = st.text_input("Buscar jugador (ej: Haaland, Rodri, Courtois)")
    hits = players[players.short_name.str.contains(q, case=False, na=False)].head(30) if q else players.sort_values("overall", ascending=False).head(30)
    name = st.selectbox("Jugador", hits.short_name + " — " + hits.player_positions)
    row = hits[(hits.short_name + " — " + hits.player_positions) == name].iloc[0]
    pred, sc = predict(row[FEATURES].values)
    real = row["y_macro"] if target == "y_macro" else row["y_fine"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Posición real (1ª listada)", real); c2.metric("Predicción SVM", pred)
    c3.metric("¿Acierta?", "Sí" if pred == real else "No")
    st.caption("Puntaje relativo de cada posición (softmax de la decisión del SVM). No es una probabilidad calibrada.")
    st.bar_chart(sc)

with tab2:
    st.caption("Ajusta atributos clave; el resto se fija en el promedio de la base.")
    base = players[FEATURES].mean()
    preset = st.selectbox("Plantilla", ["Promedio", "Portero", "Central", "Mediocentro defensivo", "Delantero centro", "Extremo"])
    P = {"Portero": dict(goalkeeping_diving=85, goalkeeping_handling=82, goalkeeping_reflexes=86, goalkeeping_positioning=83, attacking_finishing=12),
         "Central": dict(defending_standing_tackle=82, defending_marking_awareness=82, mentality_interceptions=80, power_strength=80, attacking_finishing=30),
         "Mediocentro defensivo": dict(defending_standing_tackle=78, mentality_interceptions=78, skill_long_passing=74, power_stamina=82, attacking_finishing=50),
         "Delantero centro": dict(attacking_finishing=86, mentality_positioning=86, power_shot_power=82, defending_standing_tackle=30),
         "Extremo": dict(skill_dribbling=86, movement_acceleration=88, movement_sprint_speed=87, attacking_crossing=75, defending_standing_tackle=35)}.get(preset, {})
    vals = base.copy()
    for k, v in P.items(): vals[k] = v
    LABELS = {
    "attacking_finishing": "Definición", "skill_dribbling": "Regate",
    "skill_long_passing": "Pase largo", "defending_standing_tackle": "Entrada (de pie)",
    "mentality_interceptions": "Intercepciones", "movement_sprint_speed": "Velocidad punta",
    "power_strength": "Fuerza", "goalkeeping_reflexes": "Reflejos de portero",
}
    keys = ["attacking_finishing", "skill_dribbling", "skill_long_passing", "defending_standing_tackle",
            "mentality_interceptions", "movement_sprint_speed", "power_strength", "goalkeeping_reflexes"]
    cols = st.columns(4)
    for i, k in enumerate(keys):
        vals[k] = cols[i % 4].slider(LABELS[k], 1, 99, int(round(vals[k])), key=f"{preset}_{k}")
    pred, sc = predict(vals.values)
    st.caption("Puntaje relativo de cada posición (softmax de la decisión del SVM). No es una probabilidad calibrada.")
    st.success(f"Predicción: **{pred}**"); st.bar_chart(sc)
