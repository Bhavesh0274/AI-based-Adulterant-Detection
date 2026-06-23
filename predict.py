"""Inference demo: load the palm-oil detector and predict on example rows."""
import os, joblib, numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__))

PALM_COLS = ["Fatty_Acid_Palmitic", "Fatty_Acid_Oleic", "Peroxide_Value",
             "Free_Fatty_Acid", "Iodine_Value", "Saponification_Value",
             "Color_Index", "Density", "Viscosity"]

model = joblib.load(os.path.join(BASE, "palm_oil_detector.joblib"))

# --- predict on a few rows from the dataset (if present) ---
csv = os.path.join(BASE, "palm_oil_adulteration_datasetsss.csv")
if os.path.exists(csv):
    pa = pd.read_csv(csv).drop(columns=["Class"])
    sample = pa.iloc[[0, 1, 2, 3, 4]][PALM_COLS]
else:
    # fallback example row if the CSV is not bundled
    sample = pd.DataFrame([{
        "Fatty_Acid_Palmitic": 44, "Fatty_Acid_Oleic": 39, "Peroxide_Value": 4,
        "Free_Fatty_Acid": 1.5, "Iodine_Value": 53, "Saponification_Value": 199,
        "Color_Index": 4, "Density": 0.912, "Viscosity": 60}])[PALM_COLS]

pred = model.predict(sample)
prob = model.predict_proba(sample)[:, 1]
print("PALM OIL detection:")
for i, (p, pr) in enumerate(zip(pred, prob)):
    print(f"  sample {i}: {'ADULTERATED' if p else 'PURE'}  (P(adulterated)={pr:.2f})")


def predict_palm(values: dict):
    """values: dict with the 9 parameter names -> (status, P(adulterated))."""
    row = pd.DataFrame([values])[PALM_COLS]
    p = model.predict(row)[0]
    pr = float(model.predict_proba(row)[:, 1][0])
    return ("ADULTERATED" if p else "PURE"), round(pr, 3)
