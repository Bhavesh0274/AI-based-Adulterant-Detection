"""
Train & evaluate the PALM OIL adulteration detector (9 physicochemical parameters).
Compares several classifiers by cross-validation, evaluates the best on a held-out
test set, saves the model, and writes a confusion-matrix figure.

Expects 'palm_oil_adulteration_datasetsss.csv' in the same folder.
"""
import os, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RS = 42
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "palm_oil_adulteration_datasetsss.csv")

print("="*72)
print("PALM OIL ADULTERATION DETECTION (9 physicochemical parameters)")
print("="*72)
pa = pd.read_csv(CSV)
X = pa.drop(columns=["Class"])
y = (pa["Class"] == "Adulterated").astype(int)        # 1 = adulterated
print(f"shape {pa.shape} | adulterated={int(y.sum())}  pure={int((y==0).sum())}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

models = {
 "LogReg":       Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]),
 "KNN":          Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
 "SVM-RBF":      Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=RS))]),
 "RandomForest": RandomForestClassifier(n_estimators=400, random_state=RS, n_jobs=-1),
 "HistGB":       HistGradientBoostingClassifier(random_state=RS),
 "XGBoost":      XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                               random_state=RS, n_jobs=-1),
}

print("\n5-fold CV on train (mean F1 / accuracy / ROC-AUC):"); print("-"*72)
res = {}
for n, m in models.items():
    sc = cross_validate(m, Xtr, ytr, cv=cv, scoring=["f1", "accuracy", "roc_auc"], n_jobs=-1)
    res[n] = (sc["test_f1"].mean(), sc["test_accuracy"].mean(), sc["test_roc_auc"].mean())
    print(f"  {n:12s}  F1={res[n][0]:.4f}  acc={res[n][1]:.4f}  AUC={res[n][2]:.4f}")
best = max(res, key=lambda k: res[k][0])
print(f"\n>>> best (by CV F1): {best}")

bm = models[best]; bm.fit(Xtr, ytr)
pred = bm.predict(Xte); proba = bm.predict_proba(Xte)[:, 1]
print("\nHELD-OUT TEST (20%):"); print("-"*72)
print(f"  accuracy : {accuracy_score(yte, pred):.4f}")
print(f"  precision: {precision_score(yte, pred):.4f}")
print(f"  recall   : {recall_score(yte, pred):.4f}")
print(f"  F1       : {f1_score(yte, pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(yte, proba):.4f}")
cm = confusion_matrix(yte, pred)
print("  confusion [rows=true Pure/Adult, cols=pred]:\n", cm)

try:
    fi = bm.feature_importances_; order = np.argsort(fi)[::-1]
    print("\n  top features:", [f"{X.columns[i]}({fi[i]:.3f})" for i in order[:5]])
except Exception:
    pass

joblib.dump(bm, os.path.join(BASE, "palm_oil_detector.joblib"))

# confusion-matrix figure
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pure", "Adult"]); ax.set_yticklabels(["Pure", "Adult"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Palm oil detection ({best})")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, int(v), ha="center", va="center")
plt.tight_layout(); plt.savefig(os.path.join(BASE, "palm_results.png"), dpi=130)

print(f"\nSaved: palm_oil_detector.joblib, palm_results.png")
print(f"SUMMARY: {best} -> test acc {accuracy_score(yte, pred):.3f}, "
      f"F1 {f1_score(yte, pred):.3f}, AUC {roc_auc_score(yte, proba):.3f}")
