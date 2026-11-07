import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import prepare as p

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = ["cover", "genre", "category"]


class Featurizer:
    def fit(self, df):
        self.counts = df.groupby("Author")["Title"].count()
        base = p.build_features(df)
        self.cats = {c: base[c].cat.categories for c in CATS}
        self.tf = {}
        self.svd = {}
        for col in ["Title", "Synopsis"]:
            tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, stop_words="english")
            A = tf.fit_transform(df[col].astype(str))
            self.tf[col] = tf
            self.svd[col] = TruncatedSVD(n_components=50, random_state=SEED).fit(A)
        return self

    def transform(self, df):
        X = p.build_features(df)
        X["author_books"] = df["Author"].map(self.counts).fillna(0).values
        for c in CATS:
            v = X[c].astype(str)
            cats = self.cats[c]
            v = v.where(v.isin(cats), "Other" if "Other" in cats else cats[0])
            X[c] = pd.Categorical(v, categories=cats)
        parts = [X]
        for col, name in [("Title", "t"), ("Synopsis", "s")]:
            Z = self.svd[col].transform(self.tf[col].transform(df[col].astype(str)))
            parts.append(pd.DataFrame(Z, columns=[f"{name}_{i}" for i in range(50)], index=X.index))
        return pd.concat(parts, axis=1)


def make_model():
    return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         random_state=SEED, verbose=-1)


def score(y_true_log, pred_log):
    return r2_score(y_true_log, pred_log), mean_absolute_error(np.expm1(y_true_log), np.expm1(pred_log))


def main(data_path, out_model=os.path.join(ROOT, "models", "model.pkl"),
         out_metrics=os.path.join(ROOT, "models", "metrics.json")):
    df = p.load_raw(data_path).reset_index(drop=True)
    y = np.log1p(df["Price"].values)
    folds = {"baseline_median": [], "lightgbm": []}
    for tr, te in GroupKFold(n_splits=5).split(df, y, df["Title"].values):
        assert not set(df.loc[tr, "Title"]) & set(df.loc[te, "Title"]), "title leakage"
        dtr, dte = df.iloc[tr], df.iloc[te]
        fz = Featurizer().fit(dtr)
        Xtr, Xte = fz.transform(dtr), fz.transform(dte)
        Xtr["author_books"] = dtr["Author"].map(fz.counts).values
        base = DummyRegressor(strategy="median").fit(Xtr, y[tr])
        folds["baseline_median"].append(score(y[te], base.predict(Xte)))
        m = make_model().fit(Xtr, y[tr])
        folds["lightgbm"].append(score(y[te], m.predict(Xte)))
    metrics = {"n_rows": int(len(df)), "cv": "GroupKFold(5) grouped by Title"}
    for k, v in folds.items():
        r2s = [a for a, _ in v]
        maes = [b for _, b in v]
        metrics[k] = {"r2_log_mean": float(np.mean(r2s)), "r2_log_std": float(np.std(r2s)),
                      "mae_rupees_mean": float(np.mean(maes)), "mae_rupees_std": float(np.std(maes))}
    fz = Featurizer().fit(df)
    X = fz.transform(df)
    X["author_books"] = df["Author"].map(fz.counts).values
    final = make_model().fit(X, y)
    joblib.dump({"model": final, "featurizer": fz, "columns": list(X.columns)}, out_model)
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import train as _self
    _self.main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Projects\book-data\train.xlsx")