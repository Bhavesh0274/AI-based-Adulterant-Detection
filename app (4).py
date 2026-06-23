"""
Palm-Oil Adulteration Detector  -  Streamlit app
Run:  streamlit run app.py
Needs in the same folder:  palm_oil_detector.joblib
"""
import os, io, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))

PALM_COLS = ["Fatty_Acid_Palmitic", "Fatty_Acid_Oleic", "Peroxide_Value",
             "Free_Fatty_Acid", "Iodine_Value", "Saponification_Value",
             "Color_Index", "Density", "Viscosity"]

# (column, min, max, default, step) - ranges from the dataset
PALM_FIELDS = [
    ("Fatty_Acid_Palmitic", 20.0, 65.0, 42.0, 0.1),
    ("Fatty_Acid_Oleic",    10.0, 80.0, 40.0, 0.1),
    ("Peroxide_Value",       0.0, 18.0,  5.0, 0.1),
    ("Free_Fatty_Acid",      0.0,  6.0,  2.0, 0.05),
    ("Iodine_Value",        45.0, 75.0, 55.0, 0.1),
    ("Saponification_Value",185.0, 215.0, 199.0, 0.5),
    ("Color_Index",          0.0, 14.0,  5.0, 0.1),
    ("Density",              0.78, 0.99, 0.912, 0.001),
    ("Viscosity",           25.0, 110.0, 65.0, 0.5),
]

@st.cache_resource
def load_model():
    p = os.path.join(BASE, "palm_oil_detector.joblib")
    return joblib.load(p) if os.path.exists(p) else None

def predict_palm(model, df):
    """df with the 9 columns -> df + Prediction + P(adulterated)."""
    X = df[PALM_COLS].astype(float)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    res = df.copy()
    res["Prediction"] = np.where(pred == 1, "Adulterated", "Pure")
    res["P(adulterated)"] = np.round(proba, 3)
    return res

def df_download(df, name):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    st.download_button("Download results CSV", buf.getvalue(), name, "text/csv")

# ================================ UI ================================
def main():
    st.set_page_config(page_title="Palm Oil Adulteration Detector",
                       page_icon="*", layout="centered")
    st.title("Palm Oil Adulteration Detector")
    st.caption("Predicts whether a palm-oil sample is Pure or Adulterated "
               "from its 9 physicochemical parameters.")

    model = load_model()
    if model is None:
        st.error("Model file 'palm_oil_detector.joblib' not found. "
                 "Place it in the same folder as app.py.")
        st.stop()

    with st.sidebar:
        st.header("Mode")
        mode = st.radio("How many samples?", ["Single sample", "Batch (upload CSV)"])
        st.markdown("---")
        st.caption("This model performs detection (Pure vs Adulterated). "
                   "The 9-parameter dataset has no percentage labels, so an "
                   "adulteration percentage is not predicted.")

    # ---------------- SINGLE SAMPLE ----------------
    if mode == "Single sample":
        st.subheader("Enter the 9 physicochemical parameters")
        c1, c2 = st.columns(2)
        vals = {}
        for k, (name, lo, hi, dv, stp) in enumerate(PALM_FIELDS):
            with (c1 if k % 2 == 0 else c2):
                vals[name] = st.number_input(name.replace("_", " "),
                                             float(lo), float(hi), float(dv), float(stp))
        if st.button("Predict", type="primary"):
            row = pd.DataFrame([vals])[PALM_COLS]
            res = predict_palm(model, row)
            status = res["Prediction"].iloc[0]
            p = float(res["P(adulterated)"].iloc[0])
            if status == "Pure":
                st.success(f"Result: PURE   (probability of adulteration = {p:.0%})")
            else:
                st.error(f"Result: ADULTERATED   (probability of adulteration = {p:.0%})")
            st.progress(p)
            st.caption("The bar shows the model's confidence that the sample is adulterated.")

    # ---------------- BATCH ----------------
    else:
        st.subheader("Upload a CSV of multiple samples")
        st.caption(f"Required columns: {', '.join(PALM_COLS)} "
                   "(any extra columns such as 'Class' are ignored).")

        tmpl = pd.DataFrame(columns=PALM_COLS)
        st.download_button("Download a blank template CSV",
                           tmpl.to_csv(index=False), "template.csv", "text/csv")

        up = st.file_uploader("Choose a CSV file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.write(f"Loaded **{df.shape[0]} rows x {df.shape[1]} columns**.")
            missing = [c for c in PALM_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()
            out = predict_palm(model, df)
            front = ["Prediction", "P(adulterated)"]
            out = out[front + [c for c in out.columns if c not in front]]

            n_ad = int((out["Prediction"] == "Adulterated").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", len(out))
            m2.metric("Adulterated", n_ad)
            m3.metric("Pure", len(out) - n_ad)
            st.dataframe(out, use_container_width=True, height=380)
            df_download(out, "palm_predictions.csv")

    st.markdown("---")
    st.caption("Model: Random Forest detector trained on 9 physicochemical parameters "
               "(palmitic, oleic, peroxide value, free fatty acid, iodine value, "
               "saponification value, colour index, density, viscosity).")

if __name__ == "__main__":
    main()
