# confusionmatrix_quick.py
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns, matplotlib.pyplot as plt

BASE = r"C:\Users\aksha\OneDrive\Desktop\big data\trial"
DATA = os.path.join(BASE, "master_clean.csv")
OUT  = os.path.join(BASE, "report_figs")
os.makedirs(OUT, exist_ok=True)

MIN_PER_CLASS = 5      # drop labels with <5 samples
CAP_PER_CLASS = 300    # speed cap per class (tune 200–500)

print(f"Loading: {DATA}")
df = pd.read_csv(DATA, low_memory=False)
target = "prognosis"
y_raw = df[target].astype(str)

# 1) drop ultra-rare labels
vc = y_raw.value_counts()
keep_labels = vc[vc >= MIN_PER_CLASS].index
df = df[df[target].isin(keep_labels)].copy()
y_raw = df[target].astype(str)
print(f"[clean] classes kept: {y_raw.nunique()} | rows: {len(df)}")

# 2) cap per-class size to keep it fast
blocks = []
for lab, grp in df.groupby(target, sort=False):
    if len(grp) > CAP_PER_CLASS:
        grp = grp.sample(CAP_PER_CLASS, random_state=42)
    blocks.append(grp)
df = pd.concat(blocks, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
y_raw = df[target].astype(str)
print(f"[cap] rows after per-class cap: {len(df)} (min={df[target].value_counts().min()}, max={df[target].value_counts().max()})")

# 3) build X (numeric only; one-hot objects; map yes/no to 1/0)
X = df.drop(columns=[target])
for c in X.columns:
    if X[c].dtype == "object":
        vals = set(str(v).lower() for v in X[c].dropna().unique()[:20])
        if vals.issubset({"yes","no","true","false","0","1"}):
            X[c] = X[c].map({"yes":1,"true":1,"1":1,"no":0,"false":0,"0":0}).fillna(0).astype(float)
        else:
            X = pd.concat([X.drop(columns=[c]), pd.get_dummies(X[c].astype(str), prefix=c)], axis=1)

le = LabelEncoder()
y = le.fit_transform(y_raw)

# 4) stratified split (now safe & small)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) fast model
rf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
print("[train] RandomForest…")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n=== Classification report (macro view) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6) full confusion matrix (row-normalized)
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
cm_norm = cm / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_norm, ax=ax, cmap="Reds", cbar=True, square=True)
ax.set_title("Confusion Matrix (All Classes, row-normalized)")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
step = max(1, len(le.classes_) // 25)
ax.set_xticks(np.arange(0, len(le.classes_), step))
ax.set_yticks(np.arange(0, len(le.classes_), step))
ax.set_xticklabels(le.classes_[::step], rotation=90)
ax.set_yticklabels(le.classes_[::step])
plt.tight_layout()
p_full = os.path.join(OUT, "confusion_matrix_full_quick.png")
plt.savefig(p_full, dpi=220); plt.close()
print(f"✅ saved: {p_full}")

# 7) top-15 confusion matrix (cleaner for report)
true_names = pd.Series(y_test).map(dict(enumerate(le.classes_)))
top15 = true_names.value_counts().index[:15].tolist()
top15_idx = [np.where(le.classes_ == cls)[0][0] for cls in top15]
mask = np.isin(y_test, top15_idx)
cm15 = confusion_matrix(y_test[mask], y_pred[mask], labels=top15_idx)
cm15_norm = cm15 / cm15.sum(axis=1, keepdims=True)
cm15_norm = np.nan_to_num(cm15_norm)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm15_norm, ax=ax, cmap="Reds", cbar=True, square=True,
            xticklabels=top15, yticklabels=top15)
ax.set_title("Confusion Matrix (Top-15 Classes, row-normalized)")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=90); plt.tight_layout()
p_top = os.path.join(OUT, "confusion_matrix_top15_quick.png")
plt.savefig(p_top, dpi=240); plt.close()
print(f"✅ saved: {p_top}\nDone.")
