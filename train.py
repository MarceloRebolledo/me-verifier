import os, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from joblib import dump

# Entradas
X = np.load("reports/embeddings.npy")            # (N, 512)
# leemos labels.csv y nos quedamos con la columna label
labels = []
with open("reports/labels.csv", "r", encoding="utf-8") as f:
    next(f)  # header
    for line in f:
        # file,label
        parts = line.strip().split(",")
        labels.append(int(parts[-1]))
y = np.array(labels, dtype=np.int64)

# Split estratificado
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Normalización
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xva = scaler.transform(X_val)

# Clasificador
clf = LogisticRegression(
    max_iter=200,
    class_weight="balanced",   # ayuda si hay desbalance
    solver="lbfgs"
)
clf.fit(Xtr, y_train)

# Métricas
if hasattr(clf, "predict_proba"):
    p = clf.predict_proba(Xva)[:,1]
else:
    # rarísimo con LogReg, pero por si acaso
    from scipy.special import expit
    p = expit(clf.decision_function(Xva))

y_hat = (p >= 0.5).astype(int)
acc = accuracy_score(y_val, y_hat)
auc = roc_auc_score(y_val, p)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_hat, average="binary", zero_division=0)
cm = confusion_matrix(y_val, y_hat).tolist()

# Guardar artefactos
os.makedirs("models", exist_ok=True)
dump(clf, "models/model.joblib")
dump(scaler, "models/scaler.joblib")

os.makedirs("reports", exist_ok=True)
metrics = {
    "val_accuracy": float(acc),
    "val_auc": float(auc),
    "val_precision": float(prec),
    "val_recall": float(rec),
    "val_f1": float(f1),
    "confusion_matrix": cm,
    "n_train": int(X_train.shape[0]),
    "n_val": int(X_val.shape[0])
}
with open("reports/metrics.json","w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("✅ saved models/model.joblib, models/scaler.joblib, reports/metrics.json")
print("Metrics:", metrics)
