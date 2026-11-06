import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import train as t


def fake(n=200):
    rng = np.random.RandomState(0)
    words = ["love", "war", "code", "magic", "city", "night", "story", "dark", "light", "road"]
    titles = [f"Book {i % 40}" for i in range(n)]
    return pd.DataFrame({
        "Title": [f"{titles[i]} {words[i % 10]}" for i in range(n)],
        "Author": [f"Author {i % 25}" for i in range(n)],
        "Edition": [f"Paperback,– 1 Jan {1990 + i % 30}" for i in range(n)],
        "Reviews": [f"{3 + (i % 20) / 10} out of 5 stars" for i in range(n)],
        "Ratings": [f"{i + 1} customer reviews" for i in range(n)],
        "Synopsis": [" ".join(rng.choice(words, 30)) + f" t{i % 7}" for i in range(n)],
        "Genre": [f"Genre {i % 5}" for i in range(n)],
        "BookCategory": [f"Cat {i % 3}" for i in range(n)],
        "Price": rng.uniform(50, 800, n),
    })


def test_featurizer_same_columns_train_and_test():
    df = fake()
    tr, te = df.iloc[:150], df.iloc[150:]
    fz = t.Featurizer().fit(tr)
    a, b = fz.transform(tr), fz.transform(te)
    assert list(a.columns) == list(b.columns)
    assert a.shape[1] == 110
    assert len(b) == 50


def test_group_kfold_has_no_title_overlap():
    df = fake()
    for tr, te in GroupKFold(n_splits=5).split(df, groups=df["Title"].values):
        assert not set(df.loc[tr, "Title"]) & set(df.loc[te, "Title"])