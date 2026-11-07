# Book Price Predictor

[![CI](https://github.com/kartik-s-dev/Book-Price-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/kartik-s-dev/Book-Price-Predictor/actions/workflows/ci.yml)

Predicts the price (INR) of a book from its title, author, synopsis, cover type, year, genre, category, rating and number of ratings. Includes a small Streamlit demo.

## Results (5-fold cross-validation, grouped by Title)

| Model | R2 on log(price) | MAE (INR) |
|---|---|---|
| Baseline (always predict the median) | -0.009 | 317 |
| LightGBM + text features | 0.491 (std 0.023) | 243 (std 6.6) |

- About 23% lower error than the baseline, but the error is still large: the median book price in this data is 373 INR.
- It explains roughly half of the variation in log(price). Treat predictions as rough estimates, not exact prices.
- Folds are grouped by Title, so the same title never appears in both train and test (669 rows in the data repeat a title that appears earlier).
- Not tuned. These numbers come from one fixed LightGBM setup.
- Numbers are stored in `models/metrics.json`.

## Data

- Kaggle dataset "predict book prices" by anmolkumar (`anmolkumar/predict-book-prices`), file `train.xlsx`: 6237 rows, 9 columns.
- The dataset has no license listed, so the data is NOT included in this repo. Download it yourself from Kaggle.
- Prices are in INR (assumed, not stated by the source).

## How it works

- `src/prepare.py`: reads the Excel file, turns text fields into numbers (rating, number of ratings, year, cover type), top-30 genres, author book count.
- `src/train.py`: TF-IDF + SVD (50 components each) on Title and Synopsis, fitted only on the training fold. LightGBM trained on log(price). Runs the CV and saves the model.
- `app.py`: Streamlit demo.
- `tests/`: unit tests for the feature code and for the no-title-overlap split (they use small fake data, not the real dataset).

## Run it

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/train.py C:\path\to\train.xlsx
    streamlit run app.py
    python -m pytest -q

`models/model.pkl` is not in the repo (about 12 MB, built from the data), so run `src/train.py` first. Training takes a few minutes.

## Limits

- Only tested on this one dataset. Not deployed online.
- The tests check the feature code and the split, not the full training run.
- Author is used only as "how many books this author has in the training data".