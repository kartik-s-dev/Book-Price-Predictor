import re
import numpy as np
import pandas as pd

TOP_GENRES = 30


def _first_number(text):
    m = re.search(r"\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m.group()) if m else np.nan


def load_raw(path):
    return pd.read_excel(path)


def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["rating"] = df["Reviews"].map(_first_number)
    out["n_ratings"] = df["Ratings"].map(_first_number)
    out["log_n_ratings"] = np.log1p(out["n_ratings"].fillna(0))
    cover = df["Edition"].astype(str).str.split(",").str[0].str.strip()
    common = cover.value_counts()
    common = common[common >= 100].index
    out["cover"] = cover.where(cover.isin(common), "Other")
    year = df["Edition"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    out["year"] = year
    out["title_len"] = df["Title"].astype(str).str.len()
    out["synopsis_len"] = df["Synopsis"].astype(str).str.len()
    out["author_books"] = df.groupby("Author")["Title"].transform("count")
    top = df["Genre"].value_counts().head(TOP_GENRES).index
    out["genre"] = df["Genre"].where(df["Genre"].isin(top), "Other")
    out["category"] = df["BookCategory"]
    for c in ["cover", "genre", "category"]:
        out[c] = out[c].astype("category")
    return out