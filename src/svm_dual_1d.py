"""Ejemplo 1D de clase ("El ritmo de los datos"): reggae vs techno, resuelto por el DUAL
y contrastado con scikit-learn.  Ejecutar:  python src/svm_dual_1d.py
"""
import numpy as np
from sklearn.svm import SVC

# Datos: BPM. Reggae (y=-1): 90, 100.  Techno (y=+1): 120, 130.
x = np.array([90.0, 100.0, 120.0, 130.0])
y = np.array([-1.0, -1.0, 1.0, 1.0])

# 1) Dual a mano. Por KKT solo los vectores de soporte (x2=100, x3=120) tienen alpha>0;
#    la restriccion sum(alpha_i y_i)=0 obliga a alpha2 = alpha3 = alpha.
#    W(alpha) = 2*alpha - 0.5*alpha^2 * (x3 - x2)^2   ->   W'(alpha)=0
sv = [1, 2]
d2 = (x[2] - x[1]) ** 2                      # (120-100)^2 = 400
alpha = 2.0 / d2                             # 2 - 400*alpha = 0  -> 0.005
alphas = np.zeros(4); alphas[sv] = alpha
w = np.sum(alphas * y * x)                   # w = sum alpha_i y_i x_i
b = y[2] - w * x[2]                          # desde y_s (w x_s + b) = 1 con el SV techno
margen = 2.0 / abs(w)
frontera = -b / w

print("== Dual resuelto a mano ==")
print(f"alpha = {alpha:.4f}   w = {w:.4f}   b = {b:.4f}")
print(f"frontera = {frontera:.1f} BPM   margen = 2/|w| = {margen:.1f}")

# 2) Verificacion KKT: alpha_i [ y_i f(x_i) - 1 ] = 0 para todos los puntos
f = w * x + b
print("\n== Verificacion KKT ==")
print(" x_i  alpha_i   f(x_i)   y_i f(x_i)   alpha*(y f - 1)")
for xi, ai, fi, yi in zip(x, alphas, f, y):
    print(f"{xi:4.0f}  {ai:7.3f}  {fi:7.2f}  {yi*fi:9.2f}  {ai*(yi*fi-1):12.4f}")
assert np.allclose(alphas * (y * f - 1), 0), "KKT no se cumple"

# 3) Contraste con scikit-learn: C enorme ~ margen duro
clf = SVC(kernel="linear", C=1e6).fit(x.reshape(-1, 1), y)
w_sk, b_sk = clf.coef_[0, 0], clf.intercept_[0]
print("\n== scikit-learn SVC(kernel='linear', C=1e6) ==")
print(f"vectores de soporte (BPM): {clf.support_vectors_.ravel()}")
print(f"alpha*y (dual_coef): {clf.dual_coef_.ravel()}   w = {w_sk:.4f}   b = {b_sk:.4f}")
assert np.allclose([w, b], [w_sk, b_sk], atol=1e-3), "no coincide con scikit-learn"
print("\nCoincide con el calculo a mano: w=0.1, b=-11, frontera 110 BPM, margen 20.")

# 4) Caso de prueba de clase: pista de 115 BPM
print(f"\nf(115) = {w*115 + b:+.2f}  ->  {'Techno' if w*115 + b > 0 else 'Reggae'}")