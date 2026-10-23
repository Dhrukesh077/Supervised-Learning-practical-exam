# Deploying the House Price app on Streamlit Community Cloud (free)

## 0. Run it locally first
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py                                # http://localhost:8501
```
Use **Python 3.12** (the project was built and tested on 3.12; pandas 3 / numpy 2.4 need Python 3.11+).

## 1. Push the project to a PUBLIC GitHub repo
```bash
git init && git add . && git commit -m "House price regression + Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/house-price-regression-supervised-learning.git
git push -u origin main
```
These files MUST be in the repo (the app loads them at start-up):
`app/app.py`, `models/house_price_model.pkl`, `src/house_preprocessing.py`, `requirements.txt` (repo root),
`data/feature_defaults.json`, `results/model_comparison.csv`, `images/*.png`.
`data/train.csv` is NOT needed by the app (and is git-ignored).

## 2. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub (authorise access to the repo).
2. Click **Create app / New app** -> "Deploy a public app from GitHub".
3. Repository: `<you>/house-price-regression-supervised-learning` | Branch: `main` | Main file path: `app/app.py`.
4. Open **Advanced settings** and set **Python version = 3.12** (the default may be older and will break the pinned libraries).
5. (Optional) choose a custom subdomain, e.g. `house-price-estimator` -> https://house-price-estimator.streamlit.app
6. Click **Deploy**. The first build takes a few minutes (installs scikit-learn, xgboost, shap...). Watch the logs.
7. Paste the final URL into `README.md` (`PASTE_YOUR_APP_URL_HERE`) and push again; the app redeploys automatically on every push.

## 3. Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: house_preprocessing` | `src/house_preprocessing.py` must be in the repo (the app adds `src/` to the path itself) |
| `InconsistentVersionWarning` / pickle fails to load | Python or scikit-learn version differs: choose Python 3.12 and keep `scikit-learn==1.8.0` pinned |
| "No matching distribution for ..." | wrong Python version in Advanced settings (choose 3.12) |
| `FileNotFoundError: feature_defaults.json` | commit `data/feature_defaults.json` (only `data/train.csv` is ignored) |
| App is "asleep" | free apps sleep after inactivity; click "Wake up" (takes ~30 s) |
| Build is slow | optional: for a lighter deploy remove `jupyter`, `ipykernel`, `shap`, `seaborn` from `requirements.txt` (the app does not import them) |

## 4. Alternatives
Hugging Face Spaces (SDK = Streamlit, same files) or any server with `streamlit run app/app.py --server.port 8501` (Docker, Render, Railway).
