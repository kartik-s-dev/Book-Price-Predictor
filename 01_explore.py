import pandas as pd

# Load dataset
df = pd.read_csv("data/raw_books.csv")

print("--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- SUMMARY STATS ---")
print(df.describe(include='all'))