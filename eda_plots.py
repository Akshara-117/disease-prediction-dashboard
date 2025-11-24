# eda_plots.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE   = r"C:\Users\aksha\OneDrive\Desktop\big data\trial"
CSV    = os.path.join(BASE, "master_clean.csv")
OUTDIR = os.path.join(BASE, "eda_figs")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- load ----------
df = pd.read_csv(CSV, low_memory=False)

# ---------- SS1: Top 10 diseases (log scale so others are visible) ----------
plt.figure(figsize=(10,6))
counts = df["prognosis"].value_counts().head(10)
plt.bar(range(len(counts)), counts.values)
plt.xticks(range(len(counts)), counts.index, rotation=45, ha="right")
plt.yscale("log")
plt.ylabel("Count (log scale)")
plt.title("Top 10 Diseases in Dataset")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ss1_top10_diseases.png"))
plt.close()

# ---------- helper: parse age that may look like "[70-80)" or "55" ----------
def parse_age(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    nums = re.findall(r"\d+", s)
    if len(nums) == 2:           # interval like [70-80)
        lo, hi = map(int, nums[:2])
        return (lo + hi) / 2.0   # midpoint
    if len(nums) == 1:           # single number
        return float(nums[0])
    return np.nan

# make numeric age column if the dataset has 'age'
age_num = None
if "age" in df.columns:
    age_num = df["age"].apply(parse_age)
    # drop extreme impossible values if any
    age_num = age_num.clip(lower=0, upper=100)

# ---------- SS2: Age distribution (binned, clean ticks) ----------
if age_num is not None and age_num.notna().any():
    plt.figure(figsize=(10,6))
    bins = [0,20,40,60,80,100]
    labels = ["0–20","21–40","41–60","61–80","81–100"]
    age_bin = pd.cut(age_num, bins=bins, labels=labels, include_lowest=True)
    counts_age = age_bin.value_counts().reindex(labels)  # keep order
    plt.bar(labels, counts_age.values)
    plt.xlabel("Age group")
    plt.ylabel("Frequency")
    plt.title("Distribution of Age (Binned)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ss2_age_distribution.png"))
    plt.close()

# ---------- SS3: BP vs Diabetes (% in each BP bucket) ----------
# If BP numeric, bucket into Low/Normal/High; else use categories as-is
if "blood_pressure" in df.columns:
    bp = df["blood_pressure"].copy()
    # try numeric conversion first
    bp_num = pd.to_numeric(bp, errors="coerce")
    if bp_num.notna().sum() > 0:
        # define buckets (you can adjust thresholds if needed)
        bp_cat = pd.cut(
            bp_num, bins=[-np.inf, 90, 120, np.inf],
            labels=["Low","Normal","High"]
        )
    else:
        # keep text categories, normalize a few common spellings
        bp_cat = bp.astype(str).str.strip().str.title()
        bp_cat = bp_cat.replace({
            "Hypertension":"High",
            "Normal ": "Normal",
            "Low ": "Low"
        })

    is_diab = (df["prognosis"] == "Diabetes")
    ctab = pd.crosstab(bp_cat, is_diab, normalize="index") * 100
    # keep only the 3 main rows in nice order if present
    desired = [r for r in ["Low","Normal","High"] if r in ctab.index]
    if desired:
        ctab = ctab.loc[desired]
    ctab = ctab.rename(columns={False:"Non-Diabetic", True:"Diabetic"})

    plt.figure(figsize=(9,6))
    ctab.plot(kind="bar", stacked=True, color=["#8ecae6","#ff7f7f"], ax=plt.gca())
    plt.ylabel("Percentage")
    plt.title("Blood Pressure vs Diabetes (%)")
    plt.xlabel("Blood Pressure Category")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ss3_bp_vs_diabetes.png"))
    plt.close()

# ---------- SS4: Correlation heatmap (top informative numeric features) ----------
num_df = df.select_dtypes(include=["number"])
if not num_df.empty:
    # pick features with the largest total absolute correlation to others
    corr = num_df.corr().fillna(0)
    feature_scores = corr.abs().sum().sort_values(ascending=False)
    top_feats = feature_scores.head(20).index  # 20 most “connected” features
    plt.figure(figsize=(12,8))
    sns.heatmap(num_df[top_feats].corr(), cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap (Top 20 Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ss4_corr_heatmap.png"))
    plt.close()

print(f"✅ Saved screenshots to: {OUTDIR}")
