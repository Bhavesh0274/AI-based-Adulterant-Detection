"""
Edible-Oil Adulteration Detector  -  Streamlit app
Run:  streamlit run app.py
Needs in the same folder:
  palm_oil_detector.joblib , groundnut_atrmir_twostage.joblib
  (oil_utils.py optional - a fallback SNV is built in)
"""
import os, io, sys, types, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))

# ----- make the SNV transformer importable so the spectral model unpickles -----
try:
    from oil_utils import SNV  # noqa
except ModuleNotFoundError:
    from sklearn.base import BaseEstimator, TransformerMixin
    class SNV(BaseEstimator, TransformerMixin):
        """Standard Normal Variate: row-wise centre & scale of a spectrum."""
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = np.asarray(X, float)
            return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    _m = types.ModuleType("oil_utils"); _m.SNV = SNV
    sys.modules["oil_utils"] = _m

PALM_COLS = ["Fatty_Acid_Palmitic", "Fatty_Acid_Oleic", "Peroxide_Value",
             "Free_Fatty_Acid", "Iodine_Value", "Saponification_Value",
             "Color_Index", "Density", "Viscosity"]
# (label, min, max, default, step) from the dataset ranges
PALM_FIELDS = [
    ("Fatty_Acid_Palmitic", 20.0, 65.0, 42.0, 0.1),
    ("Fatty_Acid_Oleic",    10.0, 80.0, 40.0, 0.1),
    ("Peroxide_Value",       0.0, 18.0,  5.0, 0.1),
    ("Free_Fatty_Acid",      0.0,  6.0,  2.0, 0.05),
    ("Iodine_Value",        45.0, 75.0, 55.0, 0.1),
    ("Saponification_Value",185.0,215.0,199.0,0.5),
    ("Color_Index",          0.0, 14.0,  5.0, 0.1),
    ("Density",              0.78, 0.99, 0.912, 0.001),
    ("Viscosity",           25.0,110.0, 65.0, 0.5),
]
LEVELS = np.array([0, 6.25, 25, 50])

# ----------------------------- model loading -----------------------------
@st.cache_resource
def load_models():
    out = {}
    p = os.path.join(BASE, "palm_oil_detector.joblib")
    g = os.path.join(BASE, "groundnut_atrmir_twostage.joblib")
    out["palm"] = joblib.load(p) if os.path.exists(p) else None
    out["gn"] = joblib.load(g) if os.path.exists(g) else None
    return out

@st.cache_data
def spectral_template_cols():
    """Get the 128 wavelength column names from the bundled dataset if present."""
    f = os.path.join(BASE, "Groundnut_Oil_Adulteration_ATR-MIR.csv")
    if os.path.exists(f):
        cols = list(pd.read_csv(f, nrows=1).columns)
        return [c for c in cols if c not in ("Wavelength", "target")]
    return [f"w{i}" for i in range(128)]

# ----------------------------- prediction core -----------------------------
def predict_palm(model, df):
    """df with the 9 columns -> df + Prediction + P(adulterated)."""
    X = df[PALM_COLS].astype(float)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    res = df.copy()
    res["Prediction"] = np.where(pred == 1, "Adulterated", "Pure")
    res["P(adulterated)"] = np.round(proba, 3)
    return res

def spectral_feature_matrix(df):
    """Return a 2-D (n, 128) float array from a batch dataframe of spectra."""
    drop = [c for c in df.columns if str(c).strip().lower()
            in ("wavelength", "target", "class", "label", "id", "sample")]
    feat = df.drop(columns=drop, errors="ignore")
    feat = feat.apply(pd.to_numeric, errors="coerce")
    feat = feat.dropna(axis=1, how="all")
    return feat.values

def predict_spectral(gn, X):
    """X: (n,128) -> (labels, percents).  Stage1 detect -> Stage2 quantify."""
    det, quant = gn["detector"], gn["quantifier"]
    X = np.asarray(X, float)
    is_adult = det.predict(X) == 1
    pct = np.zeros(len(X))
    if is_adult.any():
        pct[is_adult] = np.clip(quant.predict(X[is_adult]).ravel(), 0, 50)
    labels = np.where(is_adult, "Adulterated", "Pure")
    return labels, np.round(pct, 1)

def df_download(df, name):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    st.download_button("Download results CSV", buf.getvalue(), name, "text/csv")

# ================================ UI ================================
def main():
    st.set_page_config(page_title="Oil Adulteration Detector", page_icon="*", layout="centered")
    st.title("Edible-Oil Adulteration Detector")
    st.caption("Detects whether an oil is adulterated and, for spectral data, estimates the percentage.")

    models = load_models()

    with st.sidebar:
        st.header("1. What data do you have?")
        data_type = st.radio(
            "Data type",
            ["ATR-MIR spectral (Groundnut oil)", "9 physicochemical parameters (Palm oil)"],
            help="Spectral data returns status + % adulteration. The 9-parameter data returns status only.")
        st.header("2. How many samples?")
        mode = st.radio("Mode", ["Single sample", "Batch (upload CSV)"])
        st.markdown("---")
        st.caption("Spectral model: detection + quantification.\n\n"
                   "9-parameter model: detection only (that dataset has no % labels).")

    spectral = data_type.startswith("ATR-MIR")
    model = models["gn"] if spectral else models["palm"]
    if model is None:
        st.error(f"Model file for '{data_type}' not found in this folder. "
                 "Place the .joblib files next to app.py.")
        st.stop()

    # ---------------- SINGLE SAMPLE ----------------
    if mode == "Single sample":
        if spectral:
            st.subheader("Enter one ATR-MIR spectrum (128 absorbance values)")
            cols = spectral_template_cols()
            f = os.path.join(BASE, "Groundnut_Oil_Adulteration_ATR-MIR.csv")
            prefill = ""
            if os.path.exists(f):
                ds = pd.read_csv(f)
                pick = st.selectbox("Optionally load an example from the dataset",
                                    ["(type my own below)"] +
                                    [f"row {i} - {ds['target'][i]}" for i in range(len(ds))])
                if pick != "(type my own below)":
                    i = int(pick.split()[1])
                    vals = ds.drop(columns=["Wavelength", "target"]).iloc[i].values
                    prefill = ", ".join(f"{v:.5f}" for v in vals)
            txt = st.text_area(
                "Paste 128 numbers (comma / space / newline separated)",
                value=prefill, height=140,
                placeholder="0.412, 0.418, 0.421, ...  (must be exactly 128 values)")
            if st.button("Predict", type="primary"):
                raw = [t for t in txt.replace(",", " ").split()]
                try:
                    vec = np.array([float(x) for x in raw])
                except ValueError:
                    st.error("Could not parse the numbers. Use only numeric values."); st.stop()
                if vec.size != 128:
                    st.error(f"Expected 128 values, got {vec.size}."); st.stop()
                label, pct = predict_spectral(model, vec.reshape(1, -1))
                if label[0] == "Pure":
                    st.success("Result: PURE  (0% adulteration)")
                else:
                    st.error(f"Result: ADULTERATED  ~ {pct[0]}%")
                    st.progress(min(pct[0] / 50, 1.0))
        else:
            st.subheader("Enter the 9 physicochemical parameters")
            c1, c2 = st.columns(2)
            vals = {}
            for k, (name, lo, hi, dv, stp) in enumerate(PALM_FIELDS):
                with (c1 if k % 2 == 0 else c2):
                    vals[name] = st.number_input(name.replace("_", " "), float(lo), float(hi),
                                                 float(dv), float(stp))
            if st.button("Predict", type="primary"):
                row = pd.DataFrame([vals])[PALM_COLS]
                res = predict_palm(model, row)
                status = res["Prediction"].iloc[0]; p = res["P(adulterated)"].iloc[0]
                if status == "Pure":
                    st.success(f"Result: PURE   (P(adulterated) = {p})")
                else:
                    st.error(f"Result: ADULTERATED   (P(adulterated) = {p})")
                st.info("Note: the palm-oil parameter dataset has no percentage labels, "
                        "so only Pure/Adulterated is predicted here.")

    # ---------------- BATCH ----------------
    else:
        st.subheader("Upload a CSV of multiple observations")
        if spectral:
            st.caption("Each row = one spectrum. Columns = the 128 wavelengths "
                       "(any 'Wavelength'/'target' columns are ignored automatically).")
        else:
            st.caption(f"Columns required: {', '.join(PALM_COLS)} "
                       "(extra columns such as 'Class' are ignored).")

        # template download
        if spectral:
            tcols = spectral_template_cols()
            tmpl = pd.DataFrame(columns=tcols)
        else:
            tmpl = pd.DataFrame(columns=PALM_COLS)
        st.download_button("Download a blank template CSV",
                           tmpl.to_csv(index=False), "template.csv", "text/csv")

        up = st.file_uploader("Choose a CSV file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.write(f"Loaded **{df.shape[0]} rows x {df.shape[1]} columns**.")
            try:
                if spectral:
                    X = spectral_feature_matrix(df)
                    if X.shape[1] != 128:
                        st.error(f"Expected 128 spectral columns, found {X.shape[1]} "
                                 "numeric feature columns after ignoring labels."); st.stop()
                    labels, pct = predict_spectral(model, X)
                    out = df.copy()
                    out.insert(0, "Predicted_%", pct)
                    out.insert(0, "Prediction", labels)
                else:
                    missing = [c for c in PALM_COLS if c not in df.columns]
                    if missing:
                        st.error(f"Missing required columns: {missing}"); st.stop()
                    out = predict_palm(model, df)
                    # reorder helpful cols to front
                    front = ["Prediction", "P(adulterated)"]
                    out = out[front + [c for c in out.columns if c not in front]]
            except Exception as e:
                st.error(f"Could not process the file: {e}"); st.stop()

            n_ad = (out["Prediction"] == "Adulterated").sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", len(out))
            m2.metric("Adulterated", int(n_ad))
            m3.metric("Pure", int(len(out) - n_ad))
            if spectral:
                ad = out.loc[out["Prediction"] == "Adulterated", "Predicted_%"]
                if len(ad):
                    st.metric("Mean adulteration % (of adulterated)", f"{ad.mean():.1f}%")
            st.dataframe(out, use_container_width=True, height=380)
            df_download(out, "predictions.csv")

    st.markdown("---")
    st.caption("Models: Groundnut ATR-MIR (SVM detector + SVR quantifier) | "
               "Palm physicochemical (Random Forest detector).")

if __name__ == "__main__":
    main()
