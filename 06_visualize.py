import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def plot_feature_importance(input_path="data/features.csv", target="price"):
    df = pd.read_csv(input_path)
    X = df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Plot Feature Importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Saved feature_importance.png")
    
    # Plot Predicted vs Actual
    preds = model.predict(X_test)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png")
    plt.close()
    print("Saved predicted_vs_actual.png")

if __name__ == "__main__":
    plot_feature_importance()