import os
import pandas as pd

def test_clean_data_exists():
    assert os.path.exists("data/clean_books.csv")

def test_no_nulls_in_target():
    df = pd.read_csv("data/clean_books.csv")
    assert df["price"].isnull().sum() == 0

def test_row_count_reasonable():
    df = pd.read_csv("data/clean_books.csv")
    assert len(df) > 10