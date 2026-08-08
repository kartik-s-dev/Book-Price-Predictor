import pandas as pd
import re

def clean_data(input_path="data/raw_books.csv", output_path="data/clean_books.csv"):
    df = pd.read_csv(input_path)
    
    # 1. Clean Price column: Remove currency symbols like Â£, £ and convert to float
    df['price'] = df['price'].astype(str).apply(lambda x: re.sub(r'[^0-9.]', '', x))
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Drop rows where price couldn't be parsed or price <= 0
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    
    # 2. Map string ratings ('One', 'Two', 'Three', 'Four', 'Five') to numeric (1, 2, 3, 4, 5)
    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    if 'raiting' in df.columns:
        df['rating'] = df['raiting'].map(rating_map).fillna(0).astype(int)
    elif 'rating' in df.columns:
        df['rating'] = df['rating'].map(rating_map).fillna(df['rating']).astype(float)

    # 3. Clean Availability: Extract stock quantity if available
    if "availability" in df.columns:
        df["in_stock_qty"] = df["availability"].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"Cleaned data shape: {df.shape}")
    print("\nCleaned columns and types:\n", df.dtypes)
    print("\nFirst 3 cleaned rows:\n", df.head(3))
    
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    clean_data()