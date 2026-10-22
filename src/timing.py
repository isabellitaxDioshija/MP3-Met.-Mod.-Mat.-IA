"""Benchmark de tiempo de cómputo: modelos x tamaño de entrenamiento (9 clases). Mínimo de 5 repeticiones."""
import sys, time, warnings, statistics as st
sys.path.insert(0, "src")
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from data import load, FEATURES
warnings.filterwarnings("ignore")
df = load(); X, y = df[FEATURES].values, df.y_fine.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = lambda c: Pipeline([("s", StandardScaler()), ("c", c)])
MODELS = {
    "Regresión logística": lambda: sc(LogisticRegression(max_iter=2000)),
    "k-NN (k=15)": lambda: sc(KNeighborsClassifier(15)),
    "LinearSVC (OvR)": lambda: sc(LinearSVC(C=1, max_iter=20000)),
    "SVC lineal OvO": lambda: sc(SVC(kernel="linear", C=1)),
    "SVC poly OvO": lambda: sc(SVC(kernel="poly", degree=3, C=1)),
    "SVC RBF OvO": lambda: sc(SVC(kernel="rbf", C=1)),
    "SVC RBF OvR": lambda: sc(OneVsRestClassifier(SVC(kernel="rbf", C=1))),
}
SIZES = [1000, 2500, 5000, 10000, len(Xtr)]
rows = []
for n in SIZES:
    idx = pd.Series(range(len(ytr))).sample(n, random_state=0).values if n < len(Xtr) else slice(None)
    Xs, ys = Xtr[idx], ytr[idx]
    for name, mk in MODELS.items():
        f, p = [], []
        for _ in range(5):
            m = mk(); t = time.perf_counter(); m.fit(Xs, ys); f.append(time.perf_counter() - t)
            t = time.perf_counter(); m.predict(Xte); p.append(time.perf_counter() - t)
        r = dict(n_train=n, model=name, fit_s=min(f), predict_s=min(p), predict_ms_per_player=1000 * min(p) / len(Xte))
        rows.append(r); print(r, flush=True)
d = pd.DataFrame(rows); d.to_csv("results/timing.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for name in MODELS:
    s = d[d.model == name]
    axs[0].plot(s.n_train, s.fit_s, "o-", label=name); axs[1].plot(s.n_train, s.predict_s, "o-", label=name)
for ax, t in zip(axs, ["Tiempo de entrenamiento (s)", "Tiempo de predicción, 3 670 jugadores (s)"]):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Muestras de entrenamiento"); ax.set_title(t); ax.grid(alpha=.3)
axs[0].legend(fontsize=8); import platform; fig.suptitle(f"Mínimo de 5 corridas · 1 CPU · scikit-learn", fontsize=10); fig.tight_layout(); fig.savefig("figures/timing.png", dpi=150)
print("LISTO", flush=True)