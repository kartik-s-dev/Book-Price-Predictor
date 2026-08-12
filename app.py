import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline and dataset features structure
model = joblib.load("models/best_model.pkl")
features_df = pd.read_csv("data/features.csv")

# Get exact numeric feature columns used in training
target_cols = ["price", "price_log"]
model_features = [col for col in features_df.select_dtypes(include=["number"]).columns if col not in target_cols]

st.title("📚 Book Price Predictor — Quick Tester")

# 1. Preset Test Scenarios Dropdown
presets = {
    "Custom Input (Manual Type)": {"rating": 3.0, "stock": 20, "title": "A Light in the Attic"},
    "Test 1: Low Rating vs Minimum Title (Boundary Test)": {"rating": 1.0, "stock": 0, "title": "a"},
    "Test 2: High Scarcity / Low Stock (Supply-Demand Test)": {"rating": 4.0, "stock": 1, "title": "Data Science Essentials"},
    "Test 3: High Stock / High Availability": {"rating": 4.0, "stock": 90, "title": "Data Science Essentials"},
    "Test 4: Long Descriptive Title Impact": {"rating": 3.5, "stock": 15, "title": "The Comprehensive Masterclass Guide to Modern Machine Learning Engineering and MLOps Pipelines"},
    "Test 5: Top Keywords Impact ('Pilgrim' / 'Guide')": {"rating": 5.0, "stock": 50, "title": "The Ultimate Pilgrim Guide"}
}

selected_preset = st.selectbox("🎯 Choose a Test Question / Scenario:", list(presets.keys()))
default_vals = presets[selected_preset]

st.divider()

# 2. Dynamic Input Fields (Auto-filled based on selected preset)
rating = st.slider("Rating", 1.0, 5.0, float(default_vals["rating"]), step=0.5)
in_stock_qty = st.number_input("In Stock Quantity", min_value=0, max_value=100, value=int(default_vals["stock"]))
title = st.text_input("Book Title", default_vals["title"])

if st.button("Predict Price", type="primary"):
    title_length = len(title)
    title_word_count = len(title.split())
    
    # Initialize features
    input_dict = {col: 0 for col in model_features}
    
    if "rating" in input_dict:
        input_dict["rating"] = rating
    if "in_stock_qty" in input_dict:
        input_dict["in_stock_qty"] = in_stock_qty
    if "title_length" in input_dict:
        input_dict["title_length"] = title_length
    if "title_word_count" in input_dict:
        input_dict["title_word_count"] = title_word_count
        
    input_df = pd.DataFrame([input_dict])[model_features]
    prediction = model.predict(input_df)[0]
    
    st.success(f"**Predicted Price:** £{prediction:.2f}")