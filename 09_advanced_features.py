import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.stats.outliers_influence import variance_inflation_factor

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df_clean = df[(df[column] >= lower) & (df[column] <= upper)].copy()
    removed = before - len(df_clean)
    print(f"Removed {removed} outlier rows from {column}")
    return df_clean, removed

def build_advanced_features(input_path="data/clean_books.csv", output_path="data/features.csv"):
    df = pd.read_csv(input_path)
    
    # 1. Outlier removal
    df, removed_count = remove_outliers_iqr(df, "price")
    
    # 2. Skewness check for log-transform
    price_skew = df["price"].skew()
    print(f"Price Skewness: {price_skew:.2f}")
    if price_skew > 1.0:
        df["price_log"] = np.log1p(df["price"])
        print("Log transform applied to price (price_log created)")
    else:
        print("Price is not heavily skewed; log transform skipped")
        
    # 3. Numeric basic features
    if "title" in df.columns:
        df["title_length"] = df["title"].astype(str).str.len()
        df["title_word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))
        
    if "availability" in df.columns and "in_stock_qty" not in df.columns:
        df["in_stock_qty"] = df["availability"].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

    # 4. TF-IDF Text Features on 'title'
    if "title" in df.columns:
        tfidf = TfidfVectorizer(max_features=10, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["title"].fillna(""))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
            index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)

    # 5. One-hot encoding category
    if "category" in df.columns:
        df = pd.get_dummies(df, columns=["category"], drop_first=True)
        
    # 6. Check VIF on numeric features
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["price", "price_log"], errors="ignore")
    numeric_df = numeric_df.dropna()
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]
    print("\n--- VIF Analysis ---")
    print(vif_data.sort_values("VIF", ascending=False))
    
    df.to_csv(output_path, index=False)
    print(f"\nAdvanced features saved to {output_path}. Dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    build_advanced_features()