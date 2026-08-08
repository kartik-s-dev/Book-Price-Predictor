import os
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_model_comparison(input_path="data/features.csv", target="price"):
    df = pd.read_csv(input_path)
    
    X = df.select_dtypes(include=["number"]).drop(columns=[target, "price_log"], errors="ignore")
    y = df[target]
    
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    print("--- 5-Fold Cross-Validation MAE Comparison ---")
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
        mae_mean = -scores.mean()
        mae_std = scores.std()
        results[name] = mae_mean
        print(f"{name}: MAE = {mae_mean:.2f} (± {mae_std:.2f})")
        
    print("\n--- Tuning XGBoost with Optuna ---")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    print(f"Best Tuned XGBoost MAE: {study.best_value:.2f}")
    print("Best Hyperparameters:", study.best_params)
    
    os.makedirs("models", exist_ok=True)
    best_model = XGBRegressor(**study.best_params, random_state=42, verbosity=0)
    final_pipeline = Pipeline([("preprocess", preprocessor), ("model", best_model)])
    final_pipeline.fit(X, y)
    
    joblib.dump(final_pipeline, "models/best_model.pkl")
    print("\nSaved trained best pipeline to models/best_model.pkl")

if __name__ == "__main__":
    run_model_comparison()