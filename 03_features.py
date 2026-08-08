import pandas as pd

def build_features(input_path="data/clean_books.csv", output_path="data/features.csv"):
    df = pd.read_csv(input_path)
    
    # 1. Feature: Title length (number of characters in title)
    if "title" in df.columns:
        df["title_length"] = df["title"].astype(str).str.len()
        
    # 2. Feature: Title word count
    if "title" in df.columns:
        df["title_word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))
        
    # 3. Feature: One-hot encode the 'category' column
    if "category" in df.columns:
        df = pd.get_dummies(df, columns=["category"], drop_first=True)
        
    print("Final columns in dataset:", df.columns.tolist())
    print("\nNumeric feature columns for ML:")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(numeric_cols)
    
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    build_features()