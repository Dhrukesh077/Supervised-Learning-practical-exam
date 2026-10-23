"""
house_preprocessing.py
----------------------
Reusable, picklable building blocks for the Ames House Price regression project.

Why a separate module?
    A scikit-learn Pipeline that contains custom classes can only be un-pickled if
    those classes are importable. Keeping them here (next to the notebook and the
    saved `house_price_model.pkl`) lets anyone run:

        import joblib
        pipe = joblib.load("house_price_model.pkl")
        pipe.predict(raw_dataframe_rows)          # -> predicted SalePrice in USD

Contents
    handle_missing()      Step 3.1  - missing-value rules from the exam sheet
    engineer_features()   Step 3.3  - TotalSF, HouseAge, RemodAge, HasGarage, HasPool
    encode_ordinal()      Step 3.4  - Po..Ex -> 1..5 (None -> 0) quality encoding
    HouseCleaner          sklearn transformer = the three functions above, chained
    LogTargetRegressor    wraps a regressor trained on log1p(y) and returns USD
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

# --------------------------------------------------------------------------- #
# Constants (taken from the exam sheet + the Ames data dictionary)
# --------------------------------------------------------------------------- #
MISSING_DROP_THRESHOLD = 0.80          # drop columns with > 80 % missing values

# Categorical columns where NaN literally means "feature not present"
CAT_NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# Numeric columns where NaN means "no such area / count" -> 0
NUM_ZERO_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

# Quality / condition columns -> ordinal integers (None -> 0)
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
]


# --------------------------------------------------------------------------- #
# Step-wise helpers (the notebook calls these one at a time so each step is visible)
# --------------------------------------------------------------------------- #
def find_sparse_columns(df, threshold=MISSING_DROP_THRESHOLD):
    """Columns whose share of *raw* missing values exceeds `threshold`."""
    frac = df.isna().mean()
    return sorted(frac[frac > threshold].index.tolist())


def handle_missing(df, drop_cols=()):
    """Exam rules: drop >80 % missing, categorical NaN -> 'None', count NaN -> 0.
    Remaining numeric NaN (LotFrontage, GarageYrBlt) and the 1 NaN in Electrical are
    imputed later *inside the Pipeline* (median / most-frequent) so that the fill
    values are learned from the training fold only."""
    out = df.copy()
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    for c in CAT_NONE_COLS:
        if c in out.columns:
            out[c] = out[c].fillna("None")
    for c in NUM_ZERO_COLS:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out


def engineer_features(df):
    """TotalSF, HouseAge, RemodAge, HasGarage, HasPool (ages clipped at 0)."""
    out = df.copy()
    out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    out["HouseAge"] = (out["YrSold"] - out["YearBuilt"]).clip(lower=0)
    out["RemodAge"] = (out["YrSold"] - out["YearRemodAdd"]).clip(lower=0)
    out["HasGarage"] = (out["GarageArea"] > 0).astype(int)
    out["HasPool"] = (out["PoolArea"] > 0).astype(int)
    return out


def encode_ordinal(df):
    """Po/Fa/TA/Gd/Ex -> 1..5, None -> 0. MSSubClass is a nominal code -> string."""
    out = df.copy()
    for c in ORDINAL_COLS:
        if c in out.columns:
            out[c] = out[c].map(QUALITY_MAP)
    if "MSSubClass" in out.columns:
        out["MSSubClass"] = out["MSSubClass"].astype(str)
    return out


# --------------------------------------------------------------------------- #
# sklearn-compatible wrappers
# --------------------------------------------------------------------------- #
class HouseCleaner(BaseEstimator, TransformerMixin):
    """Raw Ames rows -> cleaned, feature-engineered DataFrame.

    `fit` only learns which columns are >80 % missing (an unsupervised statistic).
    Everything else is a deterministic row-wise rule, so there is no leakage.
    """

    def __init__(self, drop_threshold=MISSING_DROP_THRESHOLD):
        self.drop_threshold = drop_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.drop_cols_ = find_sparse_columns(X, self.drop_threshold)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.drop(columns=[c for c in ("Id", "SalePrice") if c in X.columns])
        X = handle_missing(X, self.drop_cols_)
        X = engineer_features(X)
        X = encode_ordinal(X)
        return X


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Regressor fitted on y = log1p(SalePrice); `predict` returns USD via expm1."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor).fit(X, y)
        return self

    def predict(self, X):
        return np.expm1(self.regressor_.predict(X))
