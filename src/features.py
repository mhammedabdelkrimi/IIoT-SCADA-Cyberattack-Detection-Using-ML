"""Nettoyage + pré-traitement : UN SEUL Pipeline, utilisé par train, API et tests."""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

CAT = ["Protocol", "Service"]
# Colonnes jamais données au modèle (identifiants, dates, fuite de cible, constantes)
DROP = ["Scr_IP", "Des_IP", "Scr_port", "Des_port", "Date", "Timestamp",
        "class1", "class2", "class3", "Bad_checksum", "is_SYN_with_RST"]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """'-' -> NaN, retire les colonnes interdites, convertit tout en nombres (sauf CAT)."""
    df = df.copy().replace("-", np.nan)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    for c in df.columns:
        if c in CAT:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        else:
            s = df[c]
            if s.dtype == bool or not is_numeric_dtype(s):
                s = s.astype(str).str.upper().replace({"TRUE": "1", "FALSE": "0"})
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


class Cleaner(BaseEstimator, TransformerMixin):
    """Apprend la liste des colonnes à l'entraînement, la réimpose ensuite."""

    def fit(self, X, y=None):
        self.columns_ = list(clean_frame(X).columns)
        return self

    def transform(self, X):
        out = clean_frame(X).reindex(columns=self.columns_)   # colonne absente -> NaN
        for c in CAT:
            if c in out.columns:
                out[c] = out[c].astype(object)
        return out


class Prep(BaseEstimator, TransformerMixin):
    """Médiane/mode -> Yeo-Johnson (|skew|>=1) -> écrêtage 1-99 % -> RobustScaler + One-Hot.
    Tout est appris sur le Train uniquement (comme dans votre notebook)."""

    def fit(self, X, y=None):
        self.cat_ = [c for c in CAT if c in X.columns]
        self.num_ = [c for c in X.columns if c not in self.cat_]
        self.imp_num_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[self.num_])
        Xn = pd.DataFrame(self.imp_num_.transform(X[self.num_]), columns=self.num_, index=X.index)

        skew = Xn.skew()
        self.asym_ = [c for c in self.num_ if abs(skew[c]) >= 1 and Xn[c].nunique() > 2]
        self.pt_ = PowerTransformer(method="yeo-johnson", standardize=False)
        if self.asym_:
            Xn[self.asym_] = self.pt_.fit_transform(Xn[self.asym_])

        self.cont_ = [c for c in self.num_ if Xn[c].nunique() > 2]   # continues
        self.pass_ = [c for c in self.num_ if c not in self.cont_]   # binaires 0/1
        self.lo_ = Xn[self.cont_].quantile(0.01)
        self.hi_ = Xn[self.cont_].quantile(0.99)
        self.scaler_ = RobustScaler().fit(Xn[self.cont_].clip(self.lo_, self.hi_, axis=1))

        self.imp_cat_ = SimpleImputer(strategy="most_frequent").fit(X[self.cat_])
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            self.imp_cat_.transform(X[self.cat_]))
        return self

    def transform(self, X):
        Xn = pd.DataFrame(self.imp_num_.transform(X[self.num_]), columns=self.num_, index=X.index)
        if self.asym_:
            Xn[self.asym_] = self.pt_.transform(Xn[self.asym_])
        Xc = Xn[self.cont_].clip(self.lo_, self.hi_, axis=1)
        return np.hstack([
            self.scaler_.transform(Xc),
            Xn[self.pass_].to_numpy(dtype=float),
            self.ohe_.transform(self.imp_cat_.transform(X[self.cat_])),
        ])


def build_pipeline(C=100, class_weight=None, max_iter=500, tol=1e-3) -> Pipeline:
    return Pipeline([
        ("clean", Cleaner()),
        ("prep", Prep()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, tol=tol)),
    ])