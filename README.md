# MP3 · SVM para predecir la demarcación táctica (EA Sports FC 24)

**Curso:** Métodos y modelos matemáticos para IA · **Mini-Proyecto 3** · Dataset 4
**Integrantes:** [Nombre 1] · [Nombre 2] · [Nombre 3] · **Fecha:** [dd/mm/aaaa]

## Objetivo
Desarrollar un clasificador multiclase basado en SVM que prediga la demarcación táctica de futbolistas de EA Sports FC 24 a partir de sus atributos atléticos, comparando OvO vs OvR y los kernels lineal, polinomial y RBF frente a modelos base.

## Datos
- Fuente: [EA Sports FC 24 (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset), `male_players.csv`. Se usa solo la versión 24: **18 350 jugadores**. El subconjunto procesado está en `data/players_fc24.csv`.
- 36 variables (34 atributos base + altura y peso). Se excluyeron `ls`, `st`, `cb`, `gk`…, `club_position`, `nation_position` y `player_positions` para evitar fuga de la etiqueta.
- Etiqueta: primera posición listada, en 4 clases (GK/DEF/MID/FWD) y en 9 posiciones (GK, CB, FB, WB, CDM, CM, CAM, W, ST).
- Partición 80/20 estratificada (`random_state=42`). `StandardScaler` dentro de un `Pipeline` y `GridSearchCV` de C × γ con 3 pliegues, solo sobre el entrenamiento.

## Resultados (macro-F1 en prueba, 3 670 jugadores)
| Modelo | 4 clases | 9 posiciones |
|---|---|---|
| Clase mayoritaria | 0.126 | 0.034 |
| Regresión logística | 0.911 | 0.700 |
| k-NN (k=15) | 0.908 | 0.657 |
| SVC lineal | 0.913 | 0.691 |
| SVC polinomial | 0.910 | 0.652 |
| SVC RBF (C=1) | 0.929 | 0.694 |
| SVC RBF optimizado | 0.928 | 0.697 |

- RBF es el mejor kernel y **OvO supera a OvR** (0.694 vs 0.658) y entrena ~4 veces más rápido.
- Con 4 clases el SVM supera a la regresión logística; con 9 posiciones **no la supera** (0.697 vs 0.700).
- El ajuste de C y γ no mejoró el resultado en prueba; las diferencias de décimas de punto están dentro del ruido de una sola partición.
- Los laterales ofensivos (WB) casi no se detectan (F1 = 0.05, solo 566 jugadores) y los mediapuntas (CAM) llegan a 0.50. Los porteros se clasifican perfectamente.

## Tiempo de cómputo
Medido con `src/timing.py` (mínimo de 5 corridas, 14 680 muestras de entrenamiento, 9 clases) en un servidor de 1 CPU ejecutado por Claude, **no** en los computadores del equipo.

| Modelo | Entrenamiento (s) | Predicción (s) |
|---|---|---|
| Regresión logística | 0.55 | ≈ 0 |
| SVC lineal OvO | 2.15 | 0.60 |
| SVC RBF OvO | 2.55 | 1.69 |
| SVC RBF OvR | 11.61 | 3.35 |

El entrenamiento del SVC crece como n^1.7–1.8 (la regresión logística como n^1.0).

## Cómo reproducir
```bash
python -m venv .venv && source .venv/Scripts/activate   # Mac/Linux: .venv/bin/activate
python -m pip install -r requirements.txt
python src/svm_dual_1d.py            # ejemplo 1D de clase
python src/experiments.py y_macro    # 4 clases
python src/experiments.py y_fine     # 9 posiciones
python src/timing.py                 # tiempos
python -m streamlit run app/streamlit_app.py
```

## Estructura
`src/` (data, experiments, timing, svm_dual_1d) · `app/` (demo Streamlit) · `models/` · `results/` · `figures/` · `data/`

## Limitaciones
Una sola partición; etiqueta = primera posición listada; los atributos son valoraciones del videojuego; WB tiene pocos ejemplos.

