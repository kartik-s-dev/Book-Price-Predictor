import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_random_forest(input_path="data/features.csv", target="price"):
    df = pd.read_csv(input_path)
    
    X = df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Random Forest MAE: {mae:.2f}")
    print(f"Random Forest R²: {r2:.2f}")
    
    return model, mae, r2

if __name__ == "__main__":
    train_random_forest()