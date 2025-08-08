import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
try:
    model = joblib.load("shoe_rating_predictor.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load brands
try:
    df = pd.read_csv("MEN_SHOES.csv")
    brands = sorted(df['Brand_Name'].dropna().unique())
except Exception as e:
    st.warning(f"Could not load brand names from CSV: {e}")
    brands = ['ASIAN', 'Reebok', 'Puma', 'Adidas', 'Bata']

# UI
st.title("Prediksi Rating Sepatu 👟")
st.markdown("Nama: **Fajriya Hakim**")

brand = st.selectbox("Pilih Merek Sepatu", brands)
how_many_sold = st.number_input("Jumlah Terjual", min_value=0.0)
current_price = st.number_input("Harga Saat Ini", min_value=0.0)

if st.button("Prediksi Rating"):
    if model:
        input_df = pd.DataFrame({
            'Brand_Name': [brand],
            'How_Many_Sold': [how_many_sold],
            'Current_Price': [current_price]
        })

        try:
            prediction = model.predict(input_df)
            st.success(f"Prediksi Rating Sepatu: {round(prediction[0], 2)}")
        except Exception as e:
            st.error(f"Terjadi Error saat prediksi: {e}")
    else:
        st.error("Model belum dimuat dengan benar.")
