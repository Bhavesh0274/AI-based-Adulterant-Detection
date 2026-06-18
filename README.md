# Edible-Oil Adulteration Detector — Streamlit app

A web app that detects whether an oil is adulterated and, for ATR-MIR spectral data,
estimates the adulteration percentage.

- **ATR-MIR spectral (Groundnut)** → Pure/Adulterated **+ %** (SVM detector + SVR quantifier)
- **9 physicochemical parameters (Palm)** → Pure/Adulterated (Random Forest)
- Single-sample entry **and** batch CSV upload with downloadable results.

## Files
```
app.py                              # the Streamlit app
oil_utils.py                        # SNV transformer used by the spectral model
palm_oil_detector.joblib            # palm-oil detector
groundnut_atrmir_twostage.joblib    # groundnut detector + % quantifier
Groundnut_Oil_Adulteration_ATR-MIR.csv   # used for the "load example" picker & template
requirements.txt
```
Keep all files in the repository **root** (Streamlit Community Cloud runs from the root).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free so anyone can use it (Streamlit Community Cloud)

1. **Create a GitHub repo** (public is simplest) and put every file above in the repo root.
   Easiest with no git knowledge: on github.com click **New repository → Add file → Upload files**,
   drag all the files in, and **Commit**. Or with git:
   ```bash
   git init
   git add .
   git commit -m "Oil adulteration Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub (authorize it once).
3. Click **Create app → Deploy a public app from GitHub**.
4. Select your **repository**, **branch** (`main`), and set **Main file path** = `app.py`.
   Optionally set a custom subdomain so the link is memorable
   (e.g. `oil-adulteration.streamlit.app`).
5. Click **Deploy**. First build takes a few minutes; after that you get a public URL
   anyone can open and use.

Updating later: just push changes to GitHub — the live app redeploys automatically.

## Notes
- The app loads its model and CSV files using paths relative to `app.py`, so it works the same
  locally and on Community Cloud as long as everything sits in the repo root.
- `scikit-learn` is pinned to `1.8.0` (the version the models were trained with) so the saved
  models load without version warnings.
- Free Community Cloud apps sleep after a period of inactivity and wake on the next visit
  (a ~30-second cold start). That's normal.
- The palm/9-parameter dataset has no percentage labels, so that mode reports status only.
