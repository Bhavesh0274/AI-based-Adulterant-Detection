# Palm Oil Adulteration Detector — Streamlit app

A web app that predicts whether a palm-oil sample is **Pure** or **Adulterated** from its
9 physicochemical parameters (palmitic, oleic, peroxide value, free fatty acid, iodine value,
saponification value, colour index, density, viscosity).

- Single-sample entry (9 input boxes) and batch CSV upload with downloadable results.
- Model: Random Forest detector.

> Note: this dataset has only a Pure/Adulterated label (no percentage column), so the app
> reports **adulteration status only** — not an adulteration percentage.

## Files
```
app.py                            # the Streamlit app
palm_oil_detector.joblib          # trained Random Forest detector
train_models.py                   # (optional) retrain & evaluate from the CSV
predict.py                        # (optional) command-line inference demo
requirements.txt
```
Keep all files in the repository **root** (Streamlit Community Cloud runs from the root).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (Streamlit Community Cloud)
1. Create a GitHub repo and put `app.py`, `palm_oil_detector.joblib`, and `requirements.txt`
   in the repo root. (On github.com: New repository → Add file → Upload files → drag them in
   → Commit.)
2. Go to **https://share.streamlit.io**, sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Choose your **repo**, **branch = main**, **Main file path = `app.py`**. Optionally set a
   custom subdomain (e.g. `palm-oil-detector.streamlit.app`).
5. Click **Deploy**. After the first build you get a public URL anyone can open.

Pushing new commits to GitHub redeploys the app automatically.

## Retraining (optional)
Put `palm_oil_adulteration_datasetsss.csv` in the folder and run:
```bash
pip install xgboost matplotlib          # extra libs only needed for training
python train_models.py
```
This compares several classifiers, evaluates the best on a held-out test set, and re-saves
`palm_oil_detector.joblib` plus a confusion-matrix figure.

## Notes
- `scikit-learn` is pinned to `1.8.0` (the version the model was trained with) so the saved
  model loads without version warnings.
- Free Community Cloud apps sleep after inactivity and wake on the next visit (~30 s cold start).
