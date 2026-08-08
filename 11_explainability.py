import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

def explain_model(model_path="models/best_model.pkl", data_path="data/features.csv", target="price"):
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df.select_dtypes(include=["number"]).drop(columns=[target, "price_log"], errors="ignore")
    
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    
    X_transformed = preprocessor.transform(X)
    feature_names = X.columns.tolist()
    
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # TreeExplainer with feature_perturbation fix for XGBoost
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_transformed_df)
    
    # 1. Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    print("Saved shap_summary.png")
    
    # 2. Single Prediction Waterfall Explanation
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall_example.png")
    plt.close()
    print("Saved shap_waterfall_example.png")

if __name__ == "__main__":
    explain_model()