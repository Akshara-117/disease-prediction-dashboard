import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from groq import Groq   # Groq LLM client


# ---------- PATHS ----------
# Base directory = folder where this app.py lives
BASE = Path(__file__).resolve().parent

MODEL_DIR = BASE / "models"
FEAT_PATH = MODEL_DIR / "feature_cols.txt"
LE_PATH = MODEL_DIR / "label_encoder.pkl"
SPEC_MAP_PATH = BASE / "disease_specialist_map.csv"

BG_PATH = BASE / "medical.png"  # dashboard background image



# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Disease Prediction & Symptom Severity Platform",
    page_icon="🩺",
    layout="wide",
)


# ---------- BACKGROUND IMAGE ----------
def set_page_bg(image_path: str):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(255, 255, 255, 0.90) !important;
            backdrop-filter: blur(4px) !important;
            -webkit-backdrop-filter: blur(4px) !important;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0,0,0,0.06);
        }}
        p, span, label, li, .stMarkdown, h1, h2, h3, h4, h5, h6 {{
            color: #0d1a26 !important;
        }}
        .stCheckbox>div>div>span {{
            color: #0d1a26 !important;
            font-weight: 500;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if BG_PATH.exists():
    set_page_bg(str(BG_PATH))


# ---------- LOADERS ----------
@st.cache_resource
def load_schema():
    with open(FEAT_PATH, "r", encoding="utf-8") as f:
        feat_cols = [line.strip() for line in f if line.strip()]
    le = joblib.load(LE_PATH)
    return feat_cols, le


@st.cache_resource
def load_models():
    models = {}
    for name in ["DecisionTree", "RandomForest", "NaiveBayes", "SVM", "Stacking"]:
        path = os.path.join(MODEL_DIR, f"model_{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_spec_map():
    if os.path.exists(SPEC_MAP_PATH):
        try:
            return pd.read_csv(SPEC_MAP_PATH)
        except Exception:
            return None
    return None


feature_cols, le = load_schema()
models = load_models()
spec_map = load_spec_map()


# =====================================================================
#                          SEVERITY LOGIC
# =====================================================================

RESP = [
    "cough",
    "breathlessness",
    "chest_pain",
    "wheezing",
    "high_fever",
    "continuous_sneezing",
]
DERM = [
    "skin_rash",
    "itching",
    "nodal_skin_eruptions",
    "skin_peeling",
    "silver_like_dusting",
]
GI = ["vomiting", "stomach_pain", "diarrhoea", "acidity", "nausea", "loss_of_appetite"]
NEURO = [
    "headache",
    "dizziness",
    "loss_of_balance",
    "lack_of_concentration",
    "blurred_and_distorted_vision",
]
URO = ["burning_micturition", "spotting_urination", "painful_urination"]
INF = ["high_fever", "chills", "fatigue", "malaise", "sweating"]


def _get(d, c):
    return int(d.get(c, 0))


def compute_severity(row_dict):
    score = 0
    score += sum(_get(row_dict, c) for c in RESP)
    score += sum(_get(row_dict, c) for c in DERM)
    score += sum(_get(row_dict, c) for c in GI)
    score += sum(_get(row_dict, c) for c in NEURO)
    score += sum(_get(row_dict, c) for c in URO)
    score += sum(_get(row_dict, c) for c in INF)

    if score >= 9:
        return "Severe", score
    if score >= 4:
        return "Moderate", score
    return "Mild", score


# =====================================================================
#                       PREDICTION & MAPPING HELPERS
# =====================================================================

def predict_topk(model, Xrow, k=3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xrow)[0]
        idx = np.argsort(proba)[::-1][:k]
        return [(le.classes_[i], float(proba[i])) for i in idx]

    scores = model.decision_function(Xrow)
    if scores.ndim == 1:
        scores = np.vstack([1 - scores, scores]).T
    p = pd.Series(scores[0]).rank(pct=True).to_numpy()
    idx = np.argsort(p)[::-1][:k]
    return [(le.classes_[i], float(p[i])) for i in idx]


def specialist_for(disease):
    if spec_map is None:
        return "General Physician"

    df = spec_map
    if "disease" not in df.columns or "specialist" not in df.columns:
        return "General Physician"

    exact = df[df["disease"] == disease]
    if not exact.empty:
        return exact.iloc[0]["specialist"]

    loose = df[df["disease"].str.lower().str.contains(disease.lower(), na=False)]
    if not loose.empty:
        return loose.iloc[0]["specialist"]

    return "General Physician"


def visit_doctor_decision(severity, all_model_top1):
    max_top1_prob = max([p for _, p in all_model_top1], default=0.0)

    if severity == "Severe":
        return "Your symptoms look serious. Please visit a doctor immediately.", "red"

    if severity == "Moderate":
        if max_top1_prob >= 0.50:
            return "You should visit a doctor soon and get checked.", "orange"
        else:
            return (
                "Moderate symptoms, but the model is not very confident. "
                "Observe for 1–2 days and visit a doctor if they do not improve.",
                "gold",
            )

    # Mild
    if max_top1_prob >= 0.90:
        return (
            "Your symptoms look mild, but the model is quite confident about a condition. "
            "It is better to visit a doctor once and get a checkup.",
            "orange",
        )
    else:
        return (
            "Looks mild for now. Rest, drink water, and visit a doctor if the "
            "symptoms get worse or do not improve in 1–2 days.",
            "green",
        )


def find_doctors_nearby(city: str, specialty: str) -> pd.DataFrame:
    if not city:
        city = "Your City"

    data = [
        {
            "Clinic / Doctor": f"{specialty} Care Center",
            "Location": city,
            "Contact": "123-456-7890",
        },
        {
            "Clinic / Doctor": f"{specialty} Clinic & Diagnostics",
            "Location": city,
            "Contact": "987-654-3210",
        },
    ]
    return pd.DataFrame(data)


# =====================================================================
#                     GROQ LLM EXPLANATION HELPER
# =====================================================================

def llm_explanation_answer(question: str, context: dict | None = None) -> str:
    """
    Use Groq LLM (Llama-3) to explain the result.
    This is ONLY for explanations, not diagnosis or prescriptions.
    """
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    

    if not api_key or "YOUR_GROQ_API_KEY_HERE" in api_key:
        return (
            "LLM is not configured yet. Please set your Groq API key "
            "inside the code where it says YOUR_GROQ_API_KEY_HERE."
        )

    context_text = ""
    if context:
        severity = context.get("severity")
        score = context.get("score")
        disease = context.get("vote")
        spec = context.get("vote_spec")
        context_text = (
            f"Severity: {severity} (score {score}). "
            f"Predicted condition: {disease}. "
            f"Suggested doctor type: {spec}. "
        )

    system_prompt = """
    You are an educational health explainer assistant for a university project.
    The user already has a machine-learning based result about symptom severity
    and a POSSIBLE disease. Your job is to:

    - Explain what the severity level means in simple language.
    - Explain what the predicted disease is in general terms.
    - Suggest generic self-care tips (rest, water, when to see a doctor).
    - Encourage consulting a real doctor for decisions.

    VERY IMPORTANT SAFETY:
    - Do NOT make a diagnosis.
    - Do NOT prescribe medicines, dosages, or treatments.
    - Do NOT say the user definitely has a condition.
    - Use phrases like "might be", "could indicate", "this tool only gives hints".
    """

    user_message = (
        f"User question: {question}\n\n"
        f"Context from the app: {context_text}\n\n"
        "Answer in a friendly and simple way. "
        "Keep it short (2–4 paragraphs)."
    )

    try:
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=400,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM call failed: {e}. Please try again later."


# =====================================================================
#                          TOP TITLE & SIDEBAR
# =====================================================================

st.title("🧬 Disease Prediction & Symptom Severity Platform")

with st.sidebar:
    st.title("🩺 Health Check App")
    st.markdown(
        """
        This is an **educational prototype** for patients.

        - Select your symptoms  
        - See an estimated **severity**  
        - Get a **possible condition** (from ML models)  
        - Know when to **visit a doctor**

        ⚠️ This tool does **not** replace professional medical advice.
        """
    )
    st.divider()
    st.caption("Project by: Akshara • Educational ML prototype")


# =====================================================================
#                                  TABS (Intro + App)
# =====================================================================

tab_intro, tab1, tab2 = st.tabs(
    [
        "📖 Introduction",
        "🧪 Symptom Checker",
        "ℹ️ How It Works",
    ]
)

# =====================================================================
#                     INTRO TAB – LANDING PAGE
# =====================================================================
with tab_intro:
    # Big hero-style intro similar to your crime dashboard
    st.markdown("## 🧬 Comprehensive Interactive Health Check Dashboard")

    colA, colB = st.columns([2, 1])

    with colA:
        st.markdown(
            """
            **Team:** 09 

            ---
            ### 🎯 Objective  
            Build an interactive **Disease Prediction & Symptom Severity Platform** that:
            - Lets users enter symptoms and basic health values  
            - Uses **machine learning models** to suggest a *possible* condition  
            - Estimates **severity** (Mild / Moderate / Severe)  
            - Advises when it is better to **visit a doctor**  

            ---
            ### 📦 Deliverables  
            - Symptom checker interface for patients  
            - Multi-model ML pipeline (Decision Tree, Random Forest, Naive Bayes, SVM, Stacking)  
            - Visual explanations of symptom severity and body systems  
            - Optional AI assistant (LLM) for simple explanations & Q&A  
            """
        )

    with colB:
        st.markdown(
            """
            ### 👩‍💻 Created By  
            - **Akshara Guvvala**  
            - **Hruthika Goud Tigulla**  
            - **Akshitha Dubbaka**

            ---
            ### ⚠️ Disclaimer  
            This is a **university / educational project**.  
            It is **not** a medical device and should **not** be used for
            real diagnosis or emergency situations.
            """
        )

    st.markdown("---")
    st.info(
        "Use the **🧪 Symptom Checker** tab next to try the app with some example "
        "symptoms and see how the ML models behave."
    )


# =====================================================================
#                     TAB 1 – SYMPTOM CHECKER + INLINE CHATBOT
# =====================================================================
with tab1:
    st.subheader("Tell us your symptoms")
    st.caption("We will estimate severity and show a possible condition.")

    if not models:
        st.error("No ML models were found. Please check the models folder.")
    else:
        used_models = list(models.keys())

        common_syms = [
            "high_fever",
            "cough",
            "breathlessness",
            "chest_pain",
            "wheezing",
            "headache",
            "dizziness",
            "fatigue",
            "nausea",
            "vomiting",
            "diarrhoea",
            "skin_rash",
            "itching",
            "burning_micturition",
            "painful_urination",
            "sweating",
            "chills",
        ]

        c1, c2 = st.columns(2)
        inputs = {}

        with c1:
            for s in common_syms[: len(common_syms) // 2]:
                if s in feature_cols:
                    label = s.replace("_", " ").title()
                    inputs[s] = 1 if st.checkbox(label, value=False) else 0

        with c2:
            for s in common_syms[len(common_syms) // 2 :]:
                if s in feature_cols:
                    label = s.replace("_", " ").title()
                    inputs[s] = 1 if st.checkbox(label, value=False) else 0

        st.markdown("### Optional Health Values (you can skip these)")

        clinical_cols = [
            "glucose",
            "serum_creatinine",
            "serum_sodium",
            "cholesterol",
            "resting_blood_pressure",
            "oldpeak",
            "age",
        ]

        c3, c4, c5 = st.columns(3)
        for i, name in enumerate(clinical_cols):
            if name in feature_cols:
                with [c3, c4, c5][i % 3]:
                    inputs[name] = st.number_input(
                        name.replace("_", " ").title(),
                        value=0.0,
                        step=0.1,
                    )

        st.markdown("---")
        city = st.text_input(
            "Your City (optional, for demo doctor suggestions):",
            value=st.session_state.get("last_city", ""),
            help="In a real system, this could be used to fetch nearby doctors.",
        )

        # ----- BUTTON: compute + store in session -----
        if st.button("🔮 Get Result", key="btn_form"):
            row = {c: 0 for c in feature_cols}
            row.update(inputs)
            Xrow = pd.DataFrame([row])

            severity, score = compute_severity(row)

            per_model = []
            all_top1 = []

            for name in used_models:
                mdl = models.get(name)
                if mdl is None:
                    continue
                topk = predict_topk(mdl, Xrow, k=3)
                top1 = topk[0]
                spec = specialist_for(top1[0])
                per_model.append((name, topk, spec))
                all_top1.append((top1[0], top1[1]))

            if not per_model:
                st.error("No valid models loaded.")
            else:
                top1_labels = [lbl for lbl, _ in all_top1]
                vote = pd.Series(top1_labels).mode().iloc[0]
                vote_spec = specialist_for(vote)
                msg, color = visit_doctor_decision(severity, all_top1)

                context = {
                    "severity": severity,
                    "score": score,
                    "per_model": per_model,
                    "vote": vote,
                    "vote_spec": vote_spec,
                    "msg": msg,
                    "color": color,
                    "row": row,
                }
                st.session_state["last_prediction"] = context
                st.session_state["last_city"] = city.strip()

        # ----- SHOW LAST RESULT (even after rerun) -----
        context = st.session_state.get("last_prediction", None)
        if context is not None:
            severity = context["severity"]
            score = context["score"]
            vote = context["vote"]
            vote_spec = context["vote_spec"]
            msg = context["msg"]
            color = context["color"]
            row = context["row"]
            last_city = st.session_state.get("last_city", "").strip()

            st.markdown("## 🧾 Your Health Summary")
            st.markdown(
                f"""
                **1. Symptom severity:**  
                → **{severity}** (score: `{score}`)

                **2. Possible condition (from ML models):**  
                → **{vote}**

                **3. Type of doctor to consult:**  
                → **{vote_spec}**

                **4. Advice:**  
                <span style="color:{color};font-size:16px;">
                {msg}
                </span>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "This is only an educational tool. "
                "Please consult a real doctor for medical decisions."
            )

            if last_city and severity != "Mild":
                st.markdown("### 🏥 Example doctors near you (demo)")
                df_docs = find_doctors_nearby(last_city, vote_spec)
                st.dataframe(df_docs, use_container_width=True)
                st.caption(
                    "These are example entries. In a real app, this would come "
                    "from a hospital/clinic API."
                )

            # ------------ VISUALS ------------
            st.markdown("### 📊 Visual summary based on your input")

            group_data = {
                "Respiratory": sum(row.get(s, 0) for s in RESP),
                "Skin / Dermatology": sum(row.get(s, 0) for s in DERM),
                "Gastro-intestinal": sum(row.get(s, 0) for s in GI),
                "Neurology": sum(row.get(s, 0) for s in NEURO),
                "Urology": sum(row.get(s, 0) for s in URO),
                "Infection-like": sum(row.get(s, 0) for s in INF),
            }
            df_group = pd.DataFrame(
                {
                    "System": list(group_data.keys()),
                    "Active Symptoms": list(group_data.values()),
                }
            )

            c_vis1, c_vis2 = st.columns(2)

            with c_vis1:
                st.markdown("**Symptoms by body system**")
                chart_group = (
                    alt.Chart(df_group)
                    .mark_bar()
                    .encode(
                        x=alt.X("System:N", sort="-y"),
                        y="Active Symptoms:Q",
                        tooltip=["System", "Active Symptoms"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_group, use_container_width=True)

            with c_vis2:
                st.markdown("**Severity score position**")
                df_sev = pd.DataFrame(
                    {
                        "Level": ["Mild (0–3)", "Moderate (4–8)", "Severe (9+)"],
                        "Min": [0, 4, 9],
                        "Max": [3, 8, 12],
                    }
                )
                df_sev["Your Score"] = score

                chart_sev = (
                    alt.Chart(df_sev)
                    .mark_bar()
                    .encode(
                        x="Level:N",
                        y="Max:Q",
                        color=alt.condition(
                            alt.datum.Min <= score,
                            alt.value("#4C78A8"),
                            alt.value("#E0E0E0"),
                        ),
                        tooltip=["Level", "Min", "Max", "Your Score"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_sev, use_container_width=True)

            # ------------ INLINE CHATBOT ------------
            st.markdown("### 💬 Ask the AI about this result (demo)")

            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            for m in st.session_state["chat_history"]:
                if m["role"] == "user":
                    st.markdown(f"**You:** {m['content']}")
                else:
                    st.markdown(f"**AI:** {m['content']}")

            prompt = st.chat_input(
                "Ask a question (e.g., 'What does Moderate mean?' or any general doubt)"
            )

            if prompt:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": prompt}
                )
                answer = llm_explanation_answer(prompt, context=context)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer}
                )


# =====================================================================
#                     TAB 2 – EXPLANATION / REPORT
# =====================================================================
with tab2:
    st.subheader("How This App Works")

    st.markdown(
        """
        ### 1. Input Layer
        - You select symptoms as **Yes/No** (0/1 features).  
        - Optional numeric values: age, glucose, blood pressure, etc.

        ### 2. Severity Scoring
        - Symptoms grouped by body system (Respiratory, Skin, GI, Neuro, Urology, Infection).  
        - Each active symptom adds to a **severity score**.  
        - Thresholds:
          - 0–3 → **Mild**  
          - 4–8 → **Moderate**  
          - ≥ 9 → **Severe**

        ### 3. Machine Learning Models
        - Decision Tree, Random Forest, Naive Bayes, SVM, Stacking Ensemble.  
        - Each model outputs probabilities for all diseases.  
        - We take **top-3** per model and do a **majority vote** on top-1 labels.  
        - Final disease is mapped to a relevant **specialist** type.

        ### 4. LLM Usage (Chatbox on Symptom Checker tab)
        - ML models do the prediction.  
        - The small chat box is meant for a **Large Language Model (LLM)** to:
          - Explain predictions and medical terms in plain language  
          - Answer general health / dashboard questions  
          - **Not** give treatment or prescriptions

        ### 5. Safety & Limitations
        - Educational / research prototype only.  
        - Predictions can be wrong or incomplete.  
        - Always treat it as a rough hint and get a **human doctor** to confirm.
        """
    )

    if "last_prediction" in st.session_state:
        lp = st.session_state["last_prediction"]
        st.markdown("### 🔬 Latest Model Outputs (from your last run)")

        data = []
        for name, topk, spec in lp["per_model"]:
            top1_label, top1_prob = topk[0]
            row = {
                "Model": name,
                "Top-1": f"{top1_label} ({top1_prob:.2f})",
                "Top-2": (
                    f"{topk[1][0]} ({topk[1][1]:.2f})"
                    if len(topk) > 1
                    else "-"
                ),
                "Top-3": (
                    f"{topk[2][0]} ({topk[2][1]:.2f})"
                    if len(topk) > 2
                    else "-"
                ),
                "Suggested Specialist (Top-1)": spec,
            }
            data.append(row)

        st.dataframe(pd.DataFrame(data), use_container_width=True)
        st.caption(
            "This table shows how each model contributed to the final condition you saw "
            "in the Symptom Checker tab."
        )
    else:
        st.info(
            "Run the **Symptom Checker** once to see a live example of per-model outputs here."
        )

st.markdown("---")
st.caption("⚠️ Educational prototype only. Not medical advice.")
