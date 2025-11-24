import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- paths --------------------
BASE = r"C:\Users\aksha\OneDrive\Desktop\big data\trial"
IN   = os.path.join(BASE, "master_clean.csv")
OUTD = os.path.join(BASE, "report_figs")
os.makedirs(OUTD, exist_ok=True)

# -------------------- load ---------------------
# low_memory=False removes the DtypeWarning
df_raw = pd.read_csv(IN, low_memory=False)
print(f"loaded: {IN}  shape={df_raw.shape}")

# make a working copy we can modify
df = df_raw.copy()

# =========================================================
# SS1 + SS2  Missing values BEFORE/AFTER (comparable)
# =========================================================
miss_before = df.isnull().sum()
top_missing_cols = miss_before.sort_values(ascending=False).head(12).index.tolist()

print("\n=== [SS1] Missing values BEFORE (same top columns used for AFTER) ===")
ss1_before = miss_before.loc[top_missing_cols]
print(ss1_before)

# --- Imputation for AFTER proof ---
# convert obvious yes/no text to numeric where possible
BOOL_MAP = {
    "yes": 1, "y": 1, "true": 1, "t": 1, "present": 1, "pos": 1, "positive": 1, "1": 1,
    "no": 0, "n": 0, "false": 0, "f": 0, "absent": 0, "neg": 0, "negative": 0, "0": 0, "": 0
}
obj_cols = df.select_dtypes(include="object").columns
for c in obj_cols:
    s = df[c].astype(str).str.strip().str.lower()
    mapped = s.map(BOOL_MAP)
    df[c] = np.where(~mapped.isna(), mapped, df[c])

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isnull().any():
        mode_val = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

miss_after = df.isnull().sum()
ss2_after = miss_after.loc[top_missing_cols]

print("\n=== [SS2] Missing values AFTER (same columns) ===")
print(ss2_after)

# save tables
ss1_before.to_csv(os.path.join(OUTD, "SS1_missing_before.csv"), header=["missing_count"])
ss2_after.to_csv(os.path.join(OUTD, "SS2_missing_after.csv"), header=["missing_count"])

# side-by-side bar chart (before vs after) for the same columns
plt.figure(figsize=(12, 6))
idx = np.arange(len(top_missing_cols))
plt.bar(idx - 0.2, ss1_before.values, width=0.4, label="Before")
plt.bar(idx + 0.2, ss2_after.values,  width=0.4, label="After")
plt.xticks(idx, top_missing_cols, rotation=45, ha="right")
plt.title("Missing Values: Before vs After (same top columns)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTD, "SS1_SS2_missing_bar_compare.png"), dpi=180)
plt.close()

# =========================================================
# SS3 + SS4  Duplicates BEFORE/AFTER
# =========================================================
print("\n=== [SS3] Row count BEFORE drop_duplicates ===")
print(len(df_raw))

keys = [c for c in ["patient_nbr", "encounter_id"] if c in df_raw.columns]
if keys:
    df_nodup = df_raw.drop_duplicates(subset=keys)
else:
    df_nodup = df_raw.drop_duplicates()

print("\n=== [SS4] Row count AFTER drop_duplicates ===")
print(len(df_nodup))

with open(os.path.join(OUTD, "SS3_SS4_duplicates.txt"), "w") as fh:
    fh.write(f"Before: {len(df_raw)}\nAfter: {len(df_nodup)}\nDropped: {len(df_raw)-len(df_nodup)}\n")



# ----------------- BEFORE -----------------
before = df["medical_specialty"].value_counts()
print("=== [SS5] BEFORE standardization ===")
print(before.head(15))

# ----------------- STANDARDIZATION -----------------
df["medical_specialty_clean"] = (
    df["medical_specialty"].astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "")
    .str.replace("-", "")
)

after_std = df["medical_specialty_clean"].value_counts()
print("\n=== [SS6] AFTER standardization ===")
print(after_std.head(15))

# ----------------- GROUPING RARE -----------------
counts = df["medical_specialty_clean"].value_counts()
df["medical_specialty_grouped"] = df["medical_specialty_clean"].apply(
    lambda x: x if counts[x] >= 50 else "other"
)

after_group = df["medical_specialty_grouped"].value_counts()
print("\n=== [SS7] AFTER grouping rare (<50) ===")
print(after_group.head(15))

print("\nCount in 'Other':", (df["medical_specialty_grouped"]=="other").sum())

# ----------------- COMPARISON SNAPSHOT -----------------
print("\n=== COMPARISON: BEFORE vs AFTER STANDARDIZATION (top 10) ===")
comparison = pd.DataFrame({
    "Before": before.head(10).index,
    "After": after_std.head(10).index
})
print(comparison)



# ========================= Outlier Proofs (IQR + Z-score) =========================
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = r"C:\Users\aksha\OneDrive\Desktop\big data\trial"
OUTD = os.path.join(BASE, "report_figs")
os.makedirs(OUTD, exist_ok=True)

# Reload a fresh df (or reuse one you already have)
df_out = pd.read_csv(os.path.join(BASE, "master_clean.csv"), low_memory=False)

# --- pick a meaningful numeric column ---
preferred = ["blood_pressure", "cholesterol_level", "glucose", "age"]
num_cols_all = df_out.select_dtypes(include=[np.number]).columns.tolist()

chosen = None
for c in preferred:
    if c in df_out.columns:
        chosen = c
        break
if chosen is None:
    # fallback: any numeric with enough spread
    for c in num_cols_all:
        if df_out[c].nunique(dropna=True) > 30:
            chosen = c
            break

if chosen is None:
    print("\n[Outliers] No suitable numeric column found.")
else:
    # coerce to numeric just in case
    s0 = pd.to_numeric(df_out[chosen], errors="coerce")

    # ---------- BEFORE stats ----------
    n_total = s0.notna().sum()
    desc_before = s0.describe()
    print(f"\n=== [Outliers] BEFORE stats for '{chosen}' ===")
    print(desc_before)

    # save BEFORE boxplot
    plt.figure()
    plt.boxplot(s0.dropna(), vert=True)
    plt.title(f"Outliers BEFORE cleaning: {chosen}")
    plt.ylabel(chosen)
    p_before = os.path.join(OUTD, f"OUT_{chosen}_box_before.png")
    plt.savefig(p_before, dpi=180, bbox_inches="tight")
    plt.close()

    # ---------- IQR detection ----------
    q1, q3 = s0.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask_low = s0 < low
    iqr_mask_high = s0 > high
    iqr_count = (iqr_mask_low | iqr_mask_high).sum()
    iqr_pct = 100 * iqr_count / n_total if n_total else 0.0
    print(f"[IQR] low={low:.3f}  high={high:.3f}  outliers={iqr_count}  ({iqr_pct:.2f}% of non-null)")

    # ---------- Z-score detection (|z| > 3) ----------
    mu = s0.mean()
    sd = s0.std(ddof=0)
    if sd and not np.isclose(sd, 0.0):
        z = (s0 - mu) / sd
        z_mask = z.abs() > 3
        z_count = z_mask.sum()
        z_pct = 100 * z_count / n_total if n_total else 0.0
        print(f"[Z-score] mean={mu:.3f}  std={sd:.3f}  |z|>3 count={z_count}  ({z_pct:.2f}%)")
    else:
        z_count = 0
        print("[Z-score] std ~ 0 → z-score not applicable")

    # ---------- HANDLE: IQR capping ----------
    s1 = s0.clip(lower=low, upper=high)
    clipped = (s1 != s0) & s0.notna()
    clipped_n = clipped.sum()
    clipped_pct = 100 * clipped_n / n_total if n_total else 0.0
    print(f"[Handle] IQR capping clipped {clipped_n} values ({clipped_pct:.2f}% of non-null)")

    # AFTER stats
    desc_after = s1.describe()
    print(f"\n=== [Outliers] AFTER stats for '{chosen}' (post IQR cap) ===")
    print(desc_after)

    # save AFTER boxplot
    plt.figure()
    plt.boxplot(s1.dropna(), vert=True)
    plt.title(f"Outliers AFTER IQR capping: {chosen}")
    plt.ylabel(chosen)
    p_after = os.path.join(OUTD, f"OUT_{chosen}_box_after.png")
    plt.savefig(p_after, dpi=180, bbox_inches="tight")
    plt.close()

    # ---------- Save summary for report ----------
    summary = pd.DataFrame({
        "metric": [
            "non_null",
            "Q1", "Q3", "IQR", "LowFence", "HighFence",
            "IQR_outliers_count", "IQR_outliers_pct",
            "Z_outliers_count(|z|>3)", "Z_outliers_pct",
            "clipped_count", "clipped_pct"
        ],
        "value": [
            n_total,
            q1, q3, iqr, low, high,
            iqr_count, round(iqr_pct, 2),
            z_count, round((100 * z_count / n_total) if n_total else 0.0, 2),
            clipped_n, round(clipped_pct, 2)
        ]
    })
    sum_path = os.path.join(OUTD, f"OUT_{chosen}_summary.csv")
    summary.to_csv(sum_path, index=False)

    print("\n[Outliers] Saved:")
    print(" - BEFORE boxplot:", p_before)
    print(" - AFTER  boxplot:", p_after)
    print(" - Summary CSV   :", sum_path)
