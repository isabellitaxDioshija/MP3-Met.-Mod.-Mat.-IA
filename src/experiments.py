"""Experimentos SVM multiclase (OvO vs OvR, kernels, C/gamma, PCA). Guarda results/*.json y figures/*.png"""
import sys, json, time, warnings
sys.path.insert(0, "src")
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from data import load, FEATURES
warnings.filterwarnings("ignore")
SEED = 42
TARGET = sys.argv[1] if len(sys.argv) > 1 else "y_fine"
R = {"target": TARGET}
df = load()
X, y = df[FEATURES].values, df[TARGET].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["n_train"], R["n_test"], R["classes"] = len(ytr), len(yte), sorted(set(y))
R["class_counts"] = pd.Series(y).value_counts().to_dict()

def ev(name, model, Xa=Xtr, Xb=Xte, store=None):
    t = time.time(); model.fit(Xa, ytr); tf = time.time() - t
    t = time.time(); p = model.predict(Xb); tp = time.time() - t
    r = dict(model=name, accuracy=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
             weighted_f1=f1_score(yte, p, average="weighted"), fit_s=round(tf, 2), predict_s=round(tp, 2))
    print(r, flush=True); (store if store is not None else R.setdefault("table", [])).append(r)
    return model, p

sc = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])

# 1. Baselines
ev("Baseline: clase mayoritaria", sc(DummyClassifier(strategy="most_frequent")))
ev("Baseline: Regresión logística", sc(LogisticRegression(max_iter=2000)))
ev("Baseline: k-NN (k=15)", sc(KNeighborsClassifier(15)))

# 2. Efecto del escalado (Regla 1)
ev("SVC RBF SIN escalar", SVC(kernel="rbf", C=1, gamma="scale"), store=R.setdefault("scaling", []))
ev("SVC RBF con StandardScaler", sc(SVC(kernel="rbf", C=1, gamma="scale")), store=R["scaling"])

# 3. Kernels (C=1) con OvO nativo
for k, kw in [("linear", {}), ("poly", {"degree": 3}), ("rbf", {})]:
    ev(f"SVC {k} OvO (C=1)", sc(SVC(kernel=k, C=1, gamma="scale", decision_function_shape="ovr", **kw)), store=R.setdefault("kernels", []))

# 4. OvO vs OvR
cmp_ = R.setdefault("ovo_ovr", [])
for k in ["linear", "rbf"]:
    ev(f"{k} OvO", sc(SVC(kernel=k, C=1, gamma="scale")), store=cmp_)
    ev(f"{k} OvR", sc(OneVsRestClassifier(SVC(kernel=k, C=1, gamma="scale"))), store=cmp_)
ev("LinearSVC OvR (liblinear)", sc(LinearSVC(C=1, max_iter=20000)), store=cmp_)

# 5. Grid conjunto C x gamma (RBF, OvO), CV estratificada 3 folds, macro-F1
pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": [0.001, 0.01, 0.1, "scale"]}
t = time.time()
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="f1_macro", n_jobs=1, return_train_score=True)
gs.fit(Xtr, ytr); R["grid_time_s"] = round(time.time() - t, 1)
R["best_params"] = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in gs.best_params_.items()}
R["best_cv_macro_f1"] = gs.best_score_
cv = pd.DataFrame(gs.cv_results_)[["param_svm__C", "param_svm__gamma", "mean_test_score", "mean_train_score"]]
cv.columns = ["C", "gamma", "cv_f1", "train_f1"]; cv.to_csv(f"results/grid_{TARGET}.csv", index=False)
print(cv, R["best_params"], flush=True)
best = gs.best_estimator_
_, pbest = ev("SVC RBF OPTIMIZADO (OvO)", best, store=R.setdefault("final", []))
R["n_support"] = int(best.named_steps["svm"].n_support_.sum()); R["n_support_by_class"] = dict(zip(best.named_steps["svm"].classes_, best.named_steps["svm"].n_support_.tolist()))
R["report"] = classification_report(yte, pbest, output_dict=True)
joblib.dump(best, f"models/svm_{TARGET}.joblib")

# 6. PCA
pca = PCA().fit(StandardScaler().fit_transform(Xtr))
cum = np.cumsum(pca.explained_variance_ratio_); R["pca_cum_var"] = cum.tolist()
R["pca_k90"] = int(np.argmax(cum >= .90) + 1); R["pca_k95"] = int(np.argmax(cum >= .95) + 1)
bp = {"C": R["best_params"]["svm__C"], "gamma": R["best_params"]["svm__gamma"]}
pca_rows = []
for k in [2, 3, 5, 10, R["pca_k95"], len(FEATURES)]:
    m = Pipeline([("s", StandardScaler()), ("p", PCA(k)), ("c", SVC(kernel="rbf", C=bp["C"], gamma=bp["gamma"]))])
    ev(f"PCA k={k} + SVC", m, store=pca_rows)
R["pca_table"] = pca_rows

# Figuras
labels = sorted(set(y)); cm = confusion_matrix(yte, pbest, labels=labels, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión normalizada ({TARGET})"); fig.tight_layout(); fig.savefig(f"figures/cm_{TARGET}.png", dpi=150); plt.close()
piv = cv.assign(gamma=cv.gamma.astype(str)).pivot(index="C", columns="gamma", values="cv_f1")
fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax); ax.set_title("Macro-F1 (CV) por C y γ"); fig.tight_layout(); fig.savefig(f"figures/grid_{TARGET}.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(range(1, len(cum) + 1), cum, "o-"); ax.axhline(.95, ls="--", c="gray"); ax.set_xlabel("Componentes"); ax.set_ylabel("Varianza acumulada"); ax.set_title("PCA"); fig.tight_layout(); fig.savefig("figures/pca_var.png", dpi=150); plt.close()
Z = Pipeline([("s", StandardScaler()), ("p", PCA(2))]).fit_transform(Xtr)
fig, ax = plt.subplots(figsize=(7, 6)); sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=ytr, s=8, alpha=.6, ax=ax); ax.set_title("Proyección PCA 2D"); fig.tight_layout(); fig.savefig(f"figures/pca2d_{TARGET}.png", dpi=150); plt.close()

json.dump(R, open(f"results/results_{TARGET}.json", "w"), indent=2, default=float)
print("LISTO", TARGET, flush=True)
