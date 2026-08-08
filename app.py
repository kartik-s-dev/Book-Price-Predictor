import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained pipeline
model = joblib.load("models/best_model.pkl")

st.title("📚 Book Price Predictor")
st.write("Enter details about the book to predict its price:")

# Input fields based on our engineered features
rating = st.slider("Rating", 1.0, 5.0, 3.0, step=0.5)
in_stock_qty = st.number_input("In Stock Quantity", min_value=0, max_value=100, value=20)
title = st.text_input("Book Title", "A Light in the Attic")

if st.button("Predict Price"):
    title_length = len(title)
    title_word_count = len(title.split())
    
    # Base feature template matching the trained model's numeric feature layout
    input_dict = {
        "rating": rating,
        "in_stock_qty": in_stock_qty,
        "title_length": title_length,
        "title_word_count": title_word_count,
        "tfidf_black": 0,
        "tfidf_condiments": 0,
        "tfidf_dark": 0,
        "tfidf_foolproof": 0,
        "tfidf_guide": 0,
        "tfidf_jellies": 0,
        "tfidf_life": 0,
        "tfidf_pickles": 0,
        "tfidf_pilgrim": 0,
        "category_Childrens": 0,
        "category_Default": 0,
        "category_Fiction": 0,
        "category_Food and Drink": 0,
        "category_Historical Fiction": 0,
        "category_History": 0,
        "category_Music": 0,
        "category_Mystery": 0,
        "category_Nonfiction": 0,
        "category_Philosophy": 0,
        "category_Poetry": 0,
        "category_Politics": 0,
        "category_Romance": 0,
        "category_Science Fiction": 0,
        "category_Sequential Art": 0,
        "category_Spirituality": 0,
        "category_Thriller": 0,
        "category_Travel": 0,
        "category_Young Adult": 0
    }
    
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    
    st.success(f"Predicted Price: £{prediction:.2f}")