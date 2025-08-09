# app.py

from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model, encoders, dan data unik yang telah disimpan
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('unique_data.pkl', 'rb') as f:
        unique_data = pickle.load(f)
except FileNotFoundError:
    print("Pastikan file 'model.pkl', 'encoders.pkl', dan 'unique_data.pkl' ada.")
    model, encoders, unique_data = None, None, None

# Route untuk halaman utama
@app.route('/')
def home():
    if not unique_data:
        return "Error: Data unik untuk form tidak ditemukan."
    # Menambahkan form_data kosong agar tidak error saat halaman pertama kali dimuat
    return render_template('index.html', unique_data=unique_data, form_data={})

# Route untuk menerima data dan memberikan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return jsonify({'error': 'Model atau encoder tidak termuat.'}), 500

    try:
        # Mengambil data dari form
        data = request.form
        
        # Membuat DataFrame dari input
        input_df = pd.DataFrame([data])
        
        # Membuat salinan untuk encoding
        input_encoded = input_df.copy()

        # Melakukan encoding pada input menggunakan encoder yang sudah disimpan
        for col in ['MenuCategory', 'MenuItem', 'Ingredients']:
            le = encoders[col]
            # Menggunakan lambda untuk menangani nilai yang mungkin tidak ada di encoder
            input_encoded[col] = input_encoded[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Mengubah tipe data harga
        input_encoded['Price'] = input_encoded['Price'].astype(float)

        # Melakukan prediksi
        prediction_encoded = model.predict(input_encoded)
        
        # Mengubah hasil prediksi dari angka kembali ke teks (High, Medium, Low)
        prediction_text = encoders['Profitability'].inverse_transform(prediction_encoded)

        # Mengembalikan hasil prediksi ke halaman web
        return render_template('index.html', 
                               unique_data=unique_data,
                               prediction_text=f'Prediksi Profitabilitas: {prediction_text[0]}',
                               form_data=data)

    except Exception as e:
        return render_template('index.html', unique_data=unique_data, prediction_text=f'Error: {str(e)}')

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
