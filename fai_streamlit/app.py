# app.py — single-page Streamlit app with tabs + presets & dropdowns

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings

# ========== General settings ==========
st.set_page_config(page_title="Adult Income — Dashboard", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ---------- Paths ----------
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------- Adult dataset categorical options ----------
CAT_OPTIONS = {
    "workclass": [
        "Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov",
        "State-gov","Without-pay","Never-worked","?"
    ],
    "education": [
        "Preschool","1st-4th","5th-6th","7th-8th","9th","10th","11th","12th",
        "HS-grad","Some-college","Assoc-voc","Assoc-acdm","Bachelors","Masters",
        "Prof-school","Doctorate"
    ],
    "marital_status": [
        "Never-married","Married-civ-spouse","Married-spouse-absent","Married-AF-spouse",
        "Divorced","Separated","Widowed"
    ],
    "occupation": [
        "Adm-clerical","Exec-managerial","Handlers-cleaners","Prof-specialty","Other-service",
        "Sales","Machine-op-inspct","Transport-moving","Farming-fishing","Tech-support",
        "Craft-repair","Protective-serv","Priv-house-serv","Armed-Forces","?"
    ],
    "relationship": [
        "Husband","Wife","Not-in-family","Own-child","Other-relative","Unmarried"
    ],
    "race": [
        "White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"
    ],
    "sex": ["Male","Female"],
    "native_country": [
        "United-States","Canada","Mexico","Puerto-Rico","Dominican-Republic","Cuba","Jamaica",
        "Guatemala","Honduras","El-Salvador","Nicaragua","Haiti","Columbia","Ecuador","Peru",
        "Brazil","Trinadad&Tobago","England","Ireland","Scotland","France","Germany","Poland",
        "Italy","Portugal","Greece","Yugoslavia","Hungary","Holand-Netherlands","India","Pakistan",
        "Bangladesh","Philippines","Vietnam","Laos","Thailand","Cambodia","Taiwan","China","Japan",
        "Hong","South","Iran","Outlying-US(Guam-USVI-etc)","?"
    ],
}

# ---------- Preset profiles (you can tweak these) ----------
PRESETS = {
    "Entry-level (private, HS-grad)":{
        "age": 23, "fnlwgt": 190000, "education_num": 9, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
        "workclass":"Private","education":"HS-grad","marital_status":"Never-married","occupation":"Adm-clerical",
        "relationship":"Not-in-family","native_country":"United-States","race":"White","sex":"Male"
    },
    "Mid-career (bachelors, married)":{
        "age": 36, "fnlwgt": 220000, "education_num": 13, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 45,
        "workclass":"Private","education":"Bachelors","marital_status":"Married-civ-spouse","occupation":"Prof-specialty",
        "relationship":"Husband","native_country":"United-States","race":"White","sex":"Male"
    },
    "Senior manager (exec)":{
        "age": 48, "fnlwgt": 250000, "education_num": 14, "capital_gain": 5000, "capital_loss": 0, "hours_per_week": 50,
        "workclass":"Private","education":"Masters","marital_status":"Married-civ-spouse","occupation":"Exec-managerial",
        "relationship":"Husband","native_country":"United-States","race":"White","sex":"Male"
    },
    "Part-time student":{
        "age": 21, "fnlwgt": 180000, "education_num": 10, "capital_gain": 0, "capital_loss": 0, "hours_per_week": 20,
        "workclass":"Private","education":"Some-college","marital_status":"Never-married","occupation":"Other-service",
        "relationship":"Own-child","native_country":"United-States","race":"Other","sex":"Female"
    }
}

# ---------- Utils ----------
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

def list_models():
    preferred = {
        "xgboost_nosensitive.pkl": "Champion (no sensitive)",
        "xgboost_tuned.pkl":       "Baseline (with sensitive)"
    }
    found = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    ordered = [f for f in preferred if f in found] + [f for f in found if f not in preferred]
    labels  = [preferred.get(f, f) for f in ordered]
    return ordered, labels

def get_raw_feature_lists(preprocessor):
    num_cols, cat_cols = [], []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name.lower().startswith("num"):
            num_cols.extend(list(cols))
        elif name.lower().startswith("cat"):
            cat_cols.extend(list(cols))
    return num_cols, cat_cols

def ensure_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in uploaded CSV: {missing}")
        return False
    return True

def apply_policy(prob: float, threshold: float, band_low: float, band_high: float) -> str:
    if prob >= band_high:  return "Positive"
    if prob >= threshold:  return "Review"
    if prob >= band_low:   return "Review"
    return "Negative"

def coerce_xgb_to_cpu(pipeline):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return pipeline
    try:
        clf = pipeline.named_steps.get("classifier", None)
        if isinstance(clf, XGBClassifier):
            try:
                clf.set_params(device="cpu", tree_method="hist", predictor="cpu_predictor")
            except Exception:
                pass
            try:
                booster = clf.get_booster()
                try: booster.set_param({"predictor": "cpu_predictor"})
                except Exception: pass
                try: booster.set_param({"device": "cpu"})
                except Exception: pass
            except Exception:
                pass
    except Exception:
        pass
    return pipeline

# ---------- Sidebar ----------
st.sidebar.title("Settings")
try:
    import sklearn, xgboost, shap, numpy, pandas
    st.sidebar.caption(
        f"sklearn {sklearn.__version__} · xgb {xgboost.__version__} · "
        f"shap {shap.__version__} · np {numpy.__version__} · pd {pandas.__version__}"
    )
except Exception:
    pass

if not os.path.isdir(MODELS_DIR):
    st.sidebar.error("Models folder not found: ./models")
    st.stop()

model_files, model_labels = list_models()
if not model_files:
    st.sidebar.error("No .pkl models found in ./models")
    st.stop()

chosen_label = st.sidebar.selectbox("Model", model_labels, index=0)
chosen_file  = model_files[model_labels.index(chosen_label)]
model_path   = os.path.join(MODELS_DIR, chosen_file)

default_threshold = 0.63 if "nosensitive" in chosen_file else 0.64
threshold  = st.sidebar.number_input("Threshold", 0.0, 1.0, default_threshold, 0.01)
band_low   = st.sidebar.number_input("Review band (low)", 0.0, 1.0, 0.55, 0.01)
band_high  = st.sidebar.number_input("Review band (high)", 0.0, 1.0, 0.75, 0.01)

pipe = load_pipeline(model_path)
pipe = coerce_xgb_to_cpu(pipe)
pre  = pipe.named_steps.get("preprocessor", None)
clf  = pipe.named_steps.get("classifier", None)

if pre is None or clf is None:
    st.sidebar.error("Selected pipeline must have `preprocessor` and `classifier` steps.")
    st.stop()

NUM_COLS, CAT_COLS = get_raw_feature_lists(pre)
expects_sensitive  = ("sex" in CAT_COLS) or ("race" in CAT_COLS)

st.sidebar.success(f"Loaded: {chosen_file}")
st.sidebar.caption(f"Expects: {len(NUM_COLS)} numeric + {len(CAT_COLS)} categorical features")

# ---------- Main ----------
st.title("Adult Income — Streamlit Dashboard")
tab_pred, tab_batch, tab_explain = st.tabs(["🔮 Predict", "📦 Batch", "🧠 Explain"])

# =========================================================
# TAB 1: Predict (single case) — with presets & dropdowns
# =========================================================
with tab_pred:
    st.subheader("Single Prediction")

    # Preset selector
    preset_name = st.selectbox("Quick preset", list(PRESETS.keys()), index=1)
    if st.button("Apply preset to form", type="primary"):
        vals = PRESETS[preset_name]
        # populate session_state so widgets pick them up
        for k,v in vals.items():
            key = f"num_{k}" if k in NUM_COLS else f"cat_{k}"
            st.session_state[key] = v
        st.experimental_rerun()

    c1, c2 = st.columns(2)
    form_vals = {}

    # Numeric widgets (sliders/inputs)
    # Ranges from the dataset description
    form_vals["age"] = c1.slider("age", 17, 90, int(st.session_state.get("num_age", PRESETS[preset_name]["age"])))
    form_vals["fnlwgt"] = c1.number_input("fnlwgt", min_value=12000, max_value=1500000,
                                          value=int(st.session_state.get("num_fnlwgt", PRESETS[preset_name]["fnlwgt"])),
                                          step=1000)
    form_vals["education_num"] = c1.slider("education_num", 1, 16,
                                           int(st.session_state.get("num_education_num", PRESETS[preset_name]["education_num"])))
    form_vals["capital_gain"] = c1.number_input("capital_gain", min_value=0, max_value=99999,
                                                value=int(st.session_state.get("num_capital_gain", PRESETS[preset_name]["capital_gain"])),
                                                step=100)
    form_vals["capital_loss"] = c1.number_input("capital_loss", min_value=0, max_value=4356,
                                                value=int(st.session_state.get("num_capital_loss", PRESETS[preset_name]["capital_loss"])),
                                                step=10)
    form_vals["hours_per_week"] = c1.slider("hours_per_week", 1, 99,
                                            int(st.session_state.get("num_hours_per_week", PRESETS[preset_name]["hours_per_week"])))

    # Categorical widgets (selectboxes)
    def cat_select(colname, default):
        choices = CAT_OPTIONS.get(colname, None)
        if choices:
            return c2.selectbox(colname, choices,
                                index=choices.index(default) if default in choices else 0,
                                key=f"cat_{colname}")
        # Fallback: text input if unknown cat
        return c2.text_input(colname, value=default, key=f"cat_{colname}")

    # Build form only for columns the model expects
    for col in CAT_COLS:
        # default from preset if available
        default_val = PRESETS[preset_name].get(col, CAT_OPTIONS.get(col, [""])[0] if CAT_OPTIONS.get(col) else "")
        form_vals[col] = cat_select(col, default_val)

    if st.button("Predict", use_container_width=True):
        # Only keep the features expected by the model (NUM_COLS + CAT_COLS)
        X = pd.DataFrame([{**{k: form_vals[k] for k in NUM_COLS},
                           **{k: form_vals[k] for k in CAT_COLS}}])

        try:
            proba = float(pipe.predict_proba(X)[0, 1])
        except Exception:
            pipe = coerce_xgb_to_cpu(pipe)
            proba = float(pipe.predict_proba(X)[0, 1])

        decision = apply_policy(proba, threshold, band_low, band_high)
        st.metric("Probability(>50K)", f"{proba:.3f}")
        st.metric("Decision", decision)
        st.caption(f"Threshold={threshold:.2f} · Review band=({band_low:.2f}, {band_high:.2f})")

# =========================================================
# TAB 2: Batch scoring
# =========================================================
with tab_batch:
    st.subheader("Batch Scoring (CSV)")
    st.caption("CSV must include the exact feature columns expected by the model.")

    template = pd.DataFrame([{**{c: 0 for c in NUM_COLS}, **{c: "" for c in CAT_COLS}}])
    st.download_button(
        "Download input template CSV",
        template.to_csv(index=False),
        file_name="adult_template.csv",
        use_container_width=True
    )

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        needed_cols = NUM_COLS + CAT_COLS
        if ensure_columns(df, needed_cols):
            try:
                probs = pipe.predict_proba(df)[:, 1]
            except Exception:
                pipe = coerce_xgb_to_cpu(pipe)
                probs = pipe.predict_proba(df)[:, 1]

            decisions = [apply_policy(p, threshold, band_low, band_high) for p in probs]
            out = df.copy()
            out["probability"] = probs
            out["decision"] = decisions
            st.dataframe(out.head(25), use_container_width=True)
            st.download_button(
                "Download predictions",
                out.to_csv(index=False),
                file_name="predictions.csv",
                use_container_width=True
            )

# =========================================================
# TAB 3: Explainability (local SHAP for one row)
# =========================================================
with tab_explain:
    st.subheader("Local SHAP (XGBoost only)")
    up_local = st.file_uploader("Upload a single-row CSV to explain", type=["csv"], key="local")
    if up_local is not None:
        df_local = pd.read_csv(up_local)
        need = NUM_COLS + CAT_COLS
        if ensure_columns(df_local, need):
            try:
                import shap
                X_local = pre.transform(df_local)
                X_bg = X_local
                expl = shap.TreeExplainer(
                    clf, data=X_bg, model_output="probability",
                    feature_perturbation="interventional"
                )
                sv = expl.shap_values(X_local)
                sv = np.asarray(sv)
                if sv.ndim == 3 and sv.shape[-1] == 2:
                    sv = sv[:, :, 1]
                base = expl.expected_value[1] if isinstance(expl.expected_value, (list, np.ndarray)) else expl.expected_value
                try:
                    feat_names = pre.get_feature_names_out()
                except Exception:
                    feat_names = np.array([f"f{i}" for i in range(X_local.shape[1])])
                exp = shap.Explanation(
                    values=sv, base_values=np.full(sv.shape[0], base),
                    data=X_local, feature_names=feat_names
                )
                st.write("Waterfall for the uploaded row:")
                shap.plots.waterfall(exp[0], max_display=15, show=False)
                st.pyplot(use_container_width=True)
            except Exception as e:
                st.warning(
                    "SHAP waterfall failed (works best with XGBoost classifiers). "
                    f"Details: {e}"
                )

# ---------- Footer ----------
st.caption(
    f"Model file: {os.path.basename(model_path)} · "
    f"Threshold={threshold:.2f} · Review band=({band_low:.2f}, {band_high:.2f})"
)
