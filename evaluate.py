import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import load

# Cargar embeddings y labels
X = np.load("reports/embeddings.npy")
labels = []
with open("reports/labels.csv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        labels.append(int(line.strip().split(",")[-1]))
y = np.array(labels, dtype=np.int64)

# Reutilizar el modelo entrenado y scaler
scaler = load("models/scaler.joblib")
clf = load("models/model.joblib")

# Validación con split nuevo (solo para graficar curvas)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
Xv = scaler.transform(X_val)
probs = clf.predict_proba(Xv)[:,1]

# Curvas
prec, rec, thr_pr = precision_recall_curve(y_val, probs)
fpr, tpr, thr_roc = roc_curve(y_val, probs)
pr_auc = auc(rec, prec)
roc_auc = auc(fpr, tpr)

# Mejor umbral según F1 aproximado
f1s = (2*prec*rec)/(prec+rec+1e-9)
best_idx = int(np.argmax(f1s))
tau = float(thr_pr[min(best_idx, len(thr_pr)-1)])
print(f"[INFO] Umbral sugerido (F1 máx): {tau:.3f}")

# Confusion matrix con ese umbral
y_hat = (probs >= tau).astype(int)
cm = confusion_matrix(y_val, y_hat).tolist()

# Guardar resultados
os.makedirs("reports", exist_ok=True)
out = {
    "tau_suggested": tau,
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "confusion_matrix": cm
}
with open("reports/threshold.json","w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

# Graficar curvas
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall curve")
plt.grid(True); plt.savefig("reports/pr_curve.png", dpi=120)
plt.close()

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
plt.grid(True); plt.savefig("reports/roc_curve.png", dpi=120)
plt.close()

print("[INFO] saved reports/threshold.json, pr_curve.png, roc_curve.png")
