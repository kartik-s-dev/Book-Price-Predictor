import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train  # noqa: F401  (needed so joblib can unpickle Featurizer)

MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")

st.set_page_config(page_title="Book Price Predictor")
st.title("Book Price Predictor")
st.caption("Trained on the MachineHack/Kaggle 'predict book prices' data (license unknown). Prices in INR.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Download train.xlsx from Kaggle (anmolkumar/predict-book-prices), "
             "then run: python src/train.py <path to train.xlsx>")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
cats = bundle["featurizer"].cats

title = st.text_input("Title", "The Prisoner's Gold (The Hunters 3)")
author = st.text_input("Author", "Chris Kuzneski")
synopsis = st.text_area("Synopsis", "A fast-paced adventure thriller.")
c1, c2 = st.columns(2)
cover = c1.selectbox("Cover", list(cats["cover"]), index=list(cats["cover"]).index("Paperback"))
year = c2.number_input("Year", 1950, 2026, 2016)
category = c1.selectbox("Category", list(cats["category"]))
genre = c2.selectbox("Genre", list(cats["genre"]))
rating = c1.slider("Rating (out of 5)", 1.0, 5.0, 4.0, 0.1)
n_ratings = c2.number_input("Number of ratings", 0, 100000, 10)

if st.button("Predict"):
    row = pd.DataFrame([{
        "Title": title,
        "Author": author,
        "Edition": f"{cover},\u2013 1 Jan {int(year)}",
        "Reviews": f"{rating} out of 5 stars",
        "Ratings": f"{int(n_ratings)} customer reviews",
        "Synopsis": synopsis,
        "Genre": genre,
        "BookCategory": category,
    }])
    X = bundle["featurizer"].transform(row)[bundle["columns"]]
    pred = float(np.expm1(bundle["model"].predict(X))[0])
    st.metric("Estimated price (INR)", f"{pred:,.0f}")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        st.caption(f"Typical error: about {m['lightgbm']['mae_rupees_mean']:.0f} INR "
                   f"(5-fold grouped CV MAE). This is a rough estimate, not an exact price.")