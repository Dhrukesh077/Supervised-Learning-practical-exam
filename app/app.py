"""Streamlit app — Ames House Price Estimator (Ridge pipeline, raw rows in -> USD out)."""
import json, pathlib, sys
import joblib, numpy as np, pandas as pd, streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]      # project root (app/ sits one level below)
sys.path.insert(0, str(ROOT / "src"))                     # house_preprocessing.py lives in src/
st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "house_price_model.pkl")      # needs src/house_preprocessing.py on the path

@st.cache_data
def load_meta():
    return json.loads((ROOT / "data" / "feature_defaults.json").read_text())

pipe, meta = load_model(), load_meta()
QUAL = ["Po", "Fa", "TA", "Gd", "Ex"]
QUAL_LABEL = {"Po": "Poor", "Fa": "Fair", "TA": "Typical/Average", "Gd": "Good", "Ex": "Excellent"}
LOG_SD = 0.113      # std-dev of hold-out residuals on the log scale (≈ ±11 %), from the notebook (results/metrics.json)

def build_row(i):
    """Start from a typical house (dataset medians/modes) and override with the user's inputs."""
    row = dict(meta["defaults"])
    row.update({"OverallQual": i["qual"], "OverallCond": i["cond"], "Neighborhood": i["nb"], "BldgType": i["bldg"],
                "HouseStyle": i["style"], "SaleCondition": i["salecond"], "LotArea": i["lot"],
                "1stFlrSF": i["f1"], "2ndFlrSF": i["f2"], "GrLivArea": i["f1"] + i["f2"],
                "YearBuilt": i["yb"], "YearRemodAdd": max(i["yr"], i["yb"]), "YrSold": i["ys"],
                "FullBath": i["fb"], "HalfBath": i["hb"], "BedroomAbvGr": i["bed"], "TotRmsAbvGrd": i["rooms"],
                "Fireplaces": i["fp"], "ExterQual": i["eq"], "KitchenQual": i["kq"]})
    # basement
    row["TotalBsmtSF"] = i["bsmt"]; row["BsmtFinSF1"] = i["bsmt"]; row["BsmtUnfSF"] = 0.0
    if i["bsmt"] == 0:
        for c in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]: row[c] = None
        row["BsmtFinSF1"] = 0.0
    else:
        row["BsmtQual"] = i["bq"]
    # garage
    row["GarageCars"] = i["gc"]; row["GarageArea"] = i["gc"] * 250.0 if i["ga"] is None else i["ga"]
    if row["GarageArea"] == 0 or i["gc"] == 0:
        row["GarageCars"] = 0; row["GarageArea"] = 0.0
        for c in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]: row[c] = None
        row["GarageYrBlt"] = None
    else:
        row["GarageYrBlt"] = float(i["yb"])
    row["FireplaceQu"] = None if i["fp"] == 0 else row["FireplaceQu"] or "Gd"
    return pd.DataFrame([row])[meta["columns"]]

def predict(row_df):
    return float(pipe.predict(row_df)[0])

# ------------------------------------------------------------------ sidebar inputs
st.sidebar.title("🏠 Describe the house")
st.sidebar.caption("Everything not listed here is set to a *typical* Ames house.")
inp = {}
with st.sidebar:
    st.subheader("Quality & location")
    inp["qual"] = st.slider("Overall quality (1 = very poor, 10 = excellent)", 1, 10, 6)
    inp["cond"] = st.slider("Overall condition (1–10)", 1, 10, 5)
    inp["nb"] = st.selectbox("Neighborhood", meta["neighborhoods"], index=meta["neighborhoods"].index("NAmes"))
    inp["eq"] = st.select_slider("Exterior quality", QUAL, "TA", format_func=QUAL_LABEL.get)
    inp["kq"] = st.select_slider("Kitchen quality", QUAL, "TA", format_func=QUAL_LABEL.get)
    st.subheader("Size")
    inp["f1"] = st.number_input("1st floor area (sq ft)", 300, 4000, 1100, 50)
    inp["f2"] = st.number_input("2nd floor area (sq ft)", 0, 2500, 0, 50)
    inp["bsmt"] = st.number_input("Basement area (sq ft, 0 = none)", 0, 3500, 1000, 50)
    inp["bq"] = st.select_slider("Basement quality", QUAL, "TA", format_func=QUAL_LABEL.get, disabled=inp["bsmt"] == 0)
    inp["lot"] = st.number_input("Lot area (sq ft)", 1000, 60000, 9500, 250)
    st.subheader("Rooms & extras")
    inp["bed"] = st.number_input("Bedrooms above ground", 0, 8, 3)
    inp["rooms"] = st.number_input("Total rooms above ground", 2, 14, 6)
    inp["fb"] = st.number_input("Full bathrooms", 0, 4, 2); inp["hb"] = st.number_input("Half bathrooms", 0, 3, 0)
    inp["fp"] = st.number_input("Fireplaces", 0, 3, 0)
    inp["gc"] = st.number_input("Garage capacity (cars, 0 = none)", 0, 4, 2)
    inp["ga"] = None
    st.subheader("Age & sale")
    inp["yb"] = st.number_input("Year built", 1872, 2010, 1990)
    inp["yr"] = st.number_input("Year remodelled (same as built if never)", 1950, 2010, 1990)
    inp["ys"] = st.number_input("Year of sale", 2006, 2010, 2008)
    inp["bldg"] = st.selectbox("Building type", meta["bldg"], index=meta["bldg"].index("1Fam"))
    inp["style"] = st.selectbox("House style", meta["style"], index=meta["style"].index("1Story"))
    inp["salecond"] = st.selectbox("Sale condition", meta["salecond"], index=meta["salecond"].index("Normal"))

# ------------------------------------------------------------------ main page
st.title("🏠 House Price Estimator")
st.caption("Ridge Regression pipeline trained on the Ames Housing dataset · Red & White Skill Education · Supervised Learning Set A")

tab1, tab2, tab3 = st.tabs(["💰 Estimate", "📊 Model performance", "ℹ️ About"])

with tab1:
    row = build_row(inp)
    price = predict(row)
    lo, hi = price * np.exp(-LOG_SD), price * np.exp(LOG_SD)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated sale price", f"${price:,.0f}")
    c2.metric("Typical range (≈ ±11 %)", f"${lo:,.0f} – ${hi:,.0f}")
    c3.metric("vs. median Ames price", f"{price / meta['median_price'] - 1:+.0%}", help=f"Dataset median: ${meta['median_price']:,.0f}")
    with st.expander("Convert to INR (optional)"):
        rate = st.number_input("USD → INR rate (edit to today's rate)", 50.0, 150.0, 85.0, 0.5)
        st.write(f"≈ **₹{price * rate / 1e5:,.1f} lakh**  (range ₹{lo * rate / 1e5:,.1f} – ₹{hi * rate / 1e5:,.1f} lakh). "
                 "Prices are 2006–2010 Ames, Iowa sales in USD — treat the INR figure as illustrative only.")
    st.subheader("How would quality change this price?")
    sweep = []
    for q in range(1, 11):
        r = row.copy(); r["OverallQual"] = q; sweep.append(predict(r))
    st.line_chart(pd.DataFrame({"Overall quality": range(1, 11), "Predicted price ($)": sweep}).set_index("Overall quality"))
    st.subheader("How would living area change this price?")
    sw2 = []
    for extra in range(-300, 1100, 100):
        f1 = max(inp["f1"] + extra, 300)
        r = row.copy(); r["1stFlrSF"] = float(f1); r["GrLivArea"] = float(f1 + inp["f2"])
        sw2.append((int(f1 + inp["f2"]), predict(r)))
    st.line_chart(pd.DataFrame(sw2, columns=["Living area (sq ft)", "Predicted price ($)"]).set_index("Living area (sq ft)"))
    with st.expander("Show the full 79-feature input row sent to the model"):
        st.dataframe(row.T.rename(columns={0: "value"}).astype(str))

with tab2:
    res = ROOT / "results" / "model_comparison.csv"
    if res.exists():
        st.subheader("Model comparison (hold-out test set, USD)")
        df = pd.read_csv(res, index_col=0)
        st.dataframe(df[["RMSE (USD)", "MAE (USD)", "R² Score", "5-Fold CV RMSE (USD)", "Training time (s)"]]
                     .style.format({"RMSE (USD)": "${:,.0f}", "MAE (USD)": "${:,.0f}", "R² Score": "{:.4f}",
                                    "5-Fold CV RMSE (USD)": "${:,.0f}", "Training time (s)": "{:.2f}"}), width="stretch")
    for f, cap in [("13_model_comparison.png", "Best model: actual vs predicted, and RMSE by model"),
                   ("14_residual_analysis.png", "Residual diagnostics"), ("16_shap_beeswarm.png", "SHAP — tuned XGBoost")]:
        p = ROOT / "images" / f
        if p.exists(): st.image(str(p), caption=cap)

with tab3:
    st.markdown("""
**What is this?** A deployed version of the exam project *House Price Prediction — Regression with Supervised Learning*.

**Model:** Ridge Regression inside one scikit-learn `Pipeline` (cleaning → feature engineering → imputation / log / scaling / encoding → Ridge).
Hold-out RMSE ≈ **$19.8k**, R² ≈ **0.93**; 5-fold CV RMSE ≈ $22.7k.

**Limits:** trained on 1,458 house sales in Ames, Iowa (2006–2010). It over-predicts very cheap, low-quality or abnormal sales, and it
should not be used for real valuation decisions in other cities or years. Inputs you do not set are filled with typical values.
""")
