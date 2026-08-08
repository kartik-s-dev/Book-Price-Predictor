import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_baseline(input_path="data/features.csv", target="price"):
    df = pd.read_csv(input_path)
    
    # Select numeric columns only and drop target from features
    X = df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Baseline MAE: {mae:.2f}")
    print(f"Baseline R²: {r2:.2f}")
    
    return model, mae, r2

if __name__ == "__main__":
    train_baseline()