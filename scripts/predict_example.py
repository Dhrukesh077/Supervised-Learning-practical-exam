"""Load the saved pipeline and predict prices for the first 5 houses in data/train.csv (raw format in, USD out).
Run from anywhere:  python scripts/predict_example.py"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                 # house_preprocessing.py is needed to un-pickle the model
import joblib, pandas as pd

pipe = joblib.load(ROOT / "models" / "house_price_model.pkl")
df = pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False, na_values=["NA"])
rows = df.head(5)
pred = pipe.predict(rows.drop(columns=["SalePrice"]))
print(pd.DataFrame({"Id": rows.Id, "Actual ($)": rows.SalePrice, "Predicted ($)": pred.round(0)}).to_string(index=False))
