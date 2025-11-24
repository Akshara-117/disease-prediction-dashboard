# merge_and_prep.py
import os
import pandas as pd
import numpy as np


# Load your merged dataset (master_clean.csv or whichever raw one you want to check)
df = pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\big data\trial\master_clean.csv")

# Show count of missing values for each column
print(df.isnull().sum().head(20))   # first 20 columns

# If you want a full view, sort columns by most missing values
print(df.isnull().sum().sort_values(ascending=False).head(10))


BASE = r"C:\Users\aksha\OneDrive\Desktop\big data\trial"

# ---------- helpers ----------
def clean_cols(df):
    df = df.copy()
    df.columns = (df.columns
                    .str.strip().str.lower()
                    .str.replace(" ", "_", regex=False)
                    .str.replace("-", "_", regex=False)
                    .str.replace("/", "_", regex=False))
    return df

def read_csv_safe(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✔ loaded: {os.path.basename(path)} -> {df.shape}")
        return clean_cols(df)
    else:
        print(f"✖ missing: {os.path.basename(path)} (skipping)")
        return None

# map common boolean-ish strings to 0/1
BOOL_MAP = {
    "yes": 1, "y": 1, "true": 1, "t": 1, "present": 1, "pos": 1, "positive": 1, "p": 1, "1": 1,
    "no": 0, "n": 0, "false": 0, "f": 0, "absent": 0, "neg": 0, "negative": 0, "0": 0, "": 0
}

def to_numeric01(s):
    """Convert a Series to numeric 0/1 if possible; otherwise coerce to 0."""
    if s.dtype.kind in "biufc":  # already numeric
        return s.fillna(0)
    # object/str → lower-string map → numeric
    s2 = s.astype(str).str.strip().str.lower()
    mapped = s2.map(BOOL_MAP)
    # where mapping missing, try numeric coercion
    if mapped.isna().any():
        coerced = pd.to_numeric(s2, errors="coerce")
        mapped = mapped.fillna(coerced)
    # final fillna → 0
    mapped = mapped.fillna(0)
    # some datasets have weird floats; clamp to 0/1
    mapped = (mapped.astype(float) > 0).astype(int)
    return mapped

SYM_GROUPS = {
    "respiratory_score": ["cough","breathlessness","chest_pain","wheezing","high_fever","continuous_sneezing"],
    "derma_score":       ["skin_rash","itching","nodal_skin_eruptions","skin_peeling","silver_like_dusting"],
    "gi_score":          ["vomiting","stomach_pain","diarrhoea","acidity","nausea","loss_of_appetite"],
    "neuro_score":       ["headache","dizziness","loss_of_balance","lack_of_concentration","blurred_and_distorted_vision"],
    "uro_score":         ["burning_micturition","spotting_urination","painful_urination"],
    "infection_severity":["high_fever","chills","fatigue","malaise","sweating"],
}

def add_symptom_scores(df):
    df = df.copy()
    newcols = {}
    for newcol, cols in SYM_GROUPS.items():
        use = [c for c in cols if c in df.columns]
        if not use:
            newcols[newcol] = 0
            print(f"[info] {newcol}: no available columns among {cols}, set to 0")
            continue
        # coerce each selected column to 0/1 numeric BEFORE summing
        tmp = pd.DataFrame({c: to_numeric01(df[c]) for c in use}, index=df.index)
        newcols[newcol] = tmp.sum(axis=1)

    # interactions (also coerced safely)
    if "high_fever" in df.columns and "skin_rash" in df.columns:
        fever = to_numeric01(df["high_fever"])
        rash  = to_numeric01(df["skin_rash"])
        newcols["fever_x_rash"] = (fever * rash).astype(int)
    else:
        newcols["fever_x_rash"] = 0

    if "fatigue" in df.columns and "weight_loss" in df.columns:
        fat   = to_numeric01(df["fatigue"])
        wloss = to_numeric01(df["weight_loss"])
        newcols["fatigue_x_weightloss"] = (fat * wloss).astype(int)
    else:
        newcols["fatigue_x_weightloss"] = 0

    # concat once to avoid fragmentation warnings
    df = pd.concat([df, pd.DataFrame(newcols, index=df.index)], axis=1)
    return df

def add_severity(df):
    def sev(row):
        s = (row.get("respiratory_score",0) + row.get("derma_score",0) + row.get("gi_score",0)
             + row.get("neuro_score",0) + row.get("uro_score",0) + row.get("infection_severity",0))
        if s >= 9: return "Severe"
        if s >= 4: return "Moderate"
        return "Mild"
    df["severity_level"] = df.apply(sev, axis=1)
    return df

# ---------- load datasets ----------
train = read_csv_safe(os.path.join(BASE, "Training.csv"))
test  = read_csv_safe(os.path.join(BASE, "Testing.csv"))
diab  = read_csv_safe(os.path.join(BASE, "diabetic_data.csv"))
heart = read_csv_safe(os.path.join(BASE, "heart_cleveland_upload.csv"))
prof  = read_csv_safe(os.path.join(BASE, "Disease_symptom_and_patient_profile.csv"))

frames = []

# 1) Symptoms Training/Test
if train is not None:
    t = train.drop_duplicates().copy()
    t["source"] = "symptoms_train"
    frames.append(t)

if test is not None and "prognosis" in test.columns:
    tt = test.drop_duplicates().copy()
    tt["source"] = "symptoms_test"
    frames.append(tt)

# 2) Patient Profile (map label to 'prognosis' if present)
if prof is not None:
    pp = prof.drop_duplicates().copy()
    for c in ["disease","prognosis","label","target"]:
        if c in pp.columns:
            if c != "prognosis":
                pp = pp.rename(columns={c: "prognosis"})
            pp["source"] = "patient_profile"
            frames.append(pp)
            break

# 3) Heart Cleveland (target/num → disease present)
if heart is not None:
    hc = heart.drop_duplicates().copy()
    pos = None
    if "target" in hc.columns:
        pos = hc[hc["target"] == 1].copy()
    elif "num" in hc.columns:
        # some variants: num > 0 indicates disease
        try:
            pos = hc[pd.to_numeric(hc["num"], errors="coerce") > 0].copy()
        except Exception:
            pos = None
    if pos is not None and len(pos) > 0:
        pos["prognosis"] = "Heart Disease"
        pos["source"] = "heart_cleveland"
        frames.append(pos)

# 4) Diabetes 130-US hospitals (all are diabetic encounters)
if diab is not None:
    dd = diab.drop_duplicates().copy()
    dd["prognosis"] = "Diabetes"
    dd["source"] = "diabetes_hosp"
    frames.append(dd)

if not frames:
    raise SystemExit("No labeled frames found. Check input files.")

# union of columns
master = pd.concat(frames, axis=0, ignore_index=True, sort=False)
print("➕ concatenated master shape:", master.shape)

if "prognosis" not in master.columns:
    raise SystemExit("master has no 'prognosis' column after merge. Cannot proceed.")

# --- One-hot encode a few common categoricals if present
cat_candidates = [c for c in ["gender","sex","race","age_group",
                              "admission_type_id","discharge_disposition_id",
                              "admission_source_id"] if c in master.columns]
master = pd.get_dummies(master, columns=cat_candidates, drop_first=True)

# --- Fill NaNs for numeric columns (pre-score)
numeric_now = [c for c in master.columns if c != "prognosis" and pd.api.types.is_numeric_dtype(master[c])]
for c in numeric_now:
    master[c] = master[c].fillna(master[c].median())

# --- Add engineered symptom scores (robust to strings)
master = add_symptom_scores(master)

# --- Severity label
master = add_severity(master)

# --- Recompute numeric columns (after new features)
numeric_now = [c for c in master.columns if c != "prognosis" and pd.api.types.is_numeric_dtype(master[c])]

# distinguish binary-like (<=2 unique values) vs continuous
binary_like = [c for c in numeric_now if master[c].dropna().nunique() <= 2]
cont_cols   = [c for c in numeric_now if c not in binary_like and master[c].dropna().nunique() > 10]

# --- Scale continuous (sklearn if available, else fallback)
if cont_cols:
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        master[cont_cols] = scaler.fit_transform(master[cont_cols])
        print(f"[scale] StandardScaler applied to {len(cont_cols)} columns")
    except Exception as e:
        master[cont_cols] = (master[cont_cols] - master[cont_cols].mean()) / master[cont_cols].std(ddof=0)
        print(f"[scale] Fallback z-score applied to {len(cont_cols)} columns (sklearn unavailable). Reason: {e}")

# final tidy
master = master.drop_duplicates().reset_index(drop=True)

# save
out_path = os.path.join(BASE, "master_clean.csv")
master.to_csv(out_path, index=False)
print(f"✅ saved: {out_path}  shape={master.shape}")

print("\nTop 10 labels:")
print(master["prognosis"].value_counts().head(10))
print("\nSample columns:", master.columns.tolist()[:30])
