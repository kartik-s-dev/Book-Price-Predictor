import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import prepare as p


def sample():
    return pd.DataFrame({
        "Title": ["A Book", "B Book", "C Book", "D Book"],
        "Author": ["X", "X", "Y", "Z"],
        "Edition": ["Paperback,– 10 Mar 2016", "Hardcover,– 7 Nov 2012", "Paperback,– 1 Jan 1990", "Paperback,– 5 May 2020"],
        "Reviews": ["4.0 out of 5 stars", "3.9 out of 5 stars", "4.8 out of 5 stars", "5.0 out of 5 stars"],
        "Ratings": ["8 customer reviews", "1,204 customer reviews", "1 customer reviews", "2 customer reviews"],
        "Synopsis": ["short text", "a bit longer text here", "x", "another synopsis"],
        "Genre": ["Romance", "Romance", "Humour", "Sports"],
        "BookCategory": ["Romance", "Romance", "Humour", "Sports"],
    })


def test_first_number_handles_commas():
    assert p._first_number("1,204 customer reviews") == 1204.0


def test_first_number_missing_is_nan():
    assert np.isnan(p._first_number("no digits"))


def test_feature_shape_and_no_price_column():
    X = p.build_features(sample())
    assert X.shape[0] == 4
    assert "Price" not in X.columns and "price" not in X.columns


def test_rating_year_and_author_count():
    X = p.build_features(sample())
    assert X["rating"].tolist() == [4.0, 3.9, 4.8, 5.0]
    assert X["year"].tolist() == [2016.0, 2012.0, 1990.0, 2020.0]
    assert X["author_books"].tolist() == [2, 2, 1, 1]