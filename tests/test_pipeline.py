import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import joblib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _load():
    pipe = joblib.load(ROOT / "models" / "house_price_model.pkl")
    df = pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False, na_values=["NA"])
    return pipe, df

def test_predicts_positive_usd_for_all_rows():
    pipe, df = _load()
    p = pipe.predict(df.drop(columns=["SalePrice"]).head(50))
    assert p.shape == (50,) and np.all(p > 10_000) and np.all(p < 1_000_000)

def test_reasonable_accuracy_on_sample():
    pipe, df = _load()
    s = df.sample(200, random_state=0)
    mape = np.mean(np.abs(pipe.predict(s.drop(columns=["SalePrice"])) / s.SalePrice - 1))
    assert mape < 0.15       # in-sample sanity check (hold-out MAPE is ~8 %)
