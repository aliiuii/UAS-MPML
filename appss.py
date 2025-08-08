import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model yang telah dilatih
try:
    model = joblib.load("shoe_rating_predictor.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Memuat data asli untuk mendapatkan daftar merek yang unik
try:
    df = pd.read_csv("MEN_SHOES.csv")
    brands = sorted(df['Brand_Name'].dropna().unique())
except Exception as e:
    print(f"Could not load brand names from CSV: {e}")
    # Daftar fallback jika file CSV tidak ditemukan
    brands = ['ASIAN', 'Reebok', 'Puma', 'Adidas', 'Bata'] 

@app.route('/')
def home():
    """
    Merender halaman utama (index.html) dengan daftar merek.
    Ganti 'NamaAnda' dengan nama Anda.
    """
    if not brands:
        return "Error: Could not load brand names for the form."
    return render_template('index.html', brands=brands, student_name="Fajriya Hakim")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Menerima input dari form, melakukan prediksi, dan mengembalikan hasil.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded!'})

    try:
        # Mengambil data dari form
        form_values = request.form.to_dict()
        
        # Membuat DataFrame dari input form.
        # Nama kolom harus sama persis dengan yang digunakan saat pelatihan.
        input_data = pd.DataFrame({
            'Brand_Name': [form_values.get('brand_name')],
            'How_Many_Sold': [float(form_values.get('how_many_sold'))],
            'Current_Price': [float(form_values.get('current_price'))]
        })

        print("Input Data received:")
        print(input_data)
        
        # Melakukan prediksi menggunakan pipeline model
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)
        
        print(f"Prediction: {output}")

        # Mengembalikan hasil prediksi ke halaman web
        # Ganti 'NamaAnda' dengan nama Anda.
        return render_template('index.html', 
                               prediction_text=f'Prediksi Rating Sepatu: {output}', 
                               brands=brands,
                               student_name="Fajriya Hakim")

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Ganti 'NamaAnda' dengan nama Anda.
        return render_template('index.html', 
                               prediction_text=f'Terjadi Error: {e}', 
                               brands=brands,
                               student_name="Fajriya Hakim")

if __name__ == "__main__":
    # Menjalankan aplikasi pada mode debug
    app.run(debug=True)
