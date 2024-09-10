from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Ini akan mengizinkan semua sumber

code_dict = {
    '10404273': {'model': r'model\lesipar.h5', 'file': r'file\lesipar.csv'},
    '10404302': {'model': r'model\metronidazol.h5', 'file': r'file\metronidazol.csv'},
    '1020410': {'model': r'model\alprazolam.h5', 'file': r'file\alprazolam.csv'},
    '10404267': {'model': r'model\lansoprazol.h5', 'file': r'file\lansoprazol.csv'},
    '10404176': {'model': r'model\digoksin.h5', 'file': r'file\digoksin.csv'},
    '1020414': {'model': r'model\merlopam.h5', 'file': r'file\merlopam.csv'},
    '10404320': {'model': r'model\nifedifin.h5', 'file': r'file\nifedifin.csv'},
    '10404374': {'model': r'model\ramipril2,5.h5', 'file': r'file\ramipril2,5.csv'},
    '1030423': {'model': r'model\amitriptilin.h5', 'file': r'file\amitriptilin.csv'},
    '1030424': {'model': r'model\arkine.h5', 'file': r'file\arkine.csv'},
    '101041': {'model': r'model\kodein10mg.h5', 'file': r'file\kodein10mg.csv'},
    '102049': {'model': r'model\alprazolam05.h5', 'file': r'file\alprazolam05.csv'},
    '10404100': {'model': r'model\sefiksim100.h5', 'file': r'file\sefiksim100.csv'},
    '10404341': {'model': r'model\parasetamol.h5', 'file': r'file\parasetamol.csv'},
    '10404372': {'model': r'model\ramipril.h5', 'file': r'file\ramipril.csv'},
    '10404102': {'model': r'model\sefiksim.h5', 'file': r'file\sefiksim.csv'},
    '10404375': {'model': r'model\ranitidin.h5', 'file': r'file\ranitidin.csv'},
    '10404110': {'model': r'model\natriumbikarbonat.h5', 'file': r'file\natriumbikarbonat.csv'},
    '10404111': {'model': r'model\misoprostol.h5', 'file': r'file\misoprostol.csv'},
    '10404132': {'model': r'model\setirizin.h5', 'file': r'file\setirizin.csv'},
    '10404139': {'model': r'model\siprofloksasin.h5', 'file': r'file\siprofloksasin.csv'},
    '10404152': {'model': r'model\klopidogrel.h5', 'file': r'file\klopidogrel.csv'},
    '10404245': {'model': r'model\isosorbid.h5', 'file': r'file\isosorbid.csv'},
    '10404292': {'model': r'model\metformin.h5', 'file': r'file\metformin.csv'},
    '10404295': {'model': r'model\metilprednisolon.h5', 'file': r'file\metilprednisolon.csv'},
    '10404312': {'model': r'model\natriumdiklofenak50.h5', 'file': r'file\natriumdiklofenak50.csv'},
    '10404363': {'model': r'model\stimox.h5', 'file': r'file\stimox.csv'},
    '10404212': {'model': r'model\furosemid.h5', 'file': r'file\furosemid.csv'},
    '10404214': {'model': r'model\gabapentin.h5', 'file': r'file\gabapentin.csv'},
    '10404318': {'model': r'model\neurohax.h5', 'file': r'file\neurohax.csv'},
    # Tambahkan lebih banyak kode sesuai kebutuhan
}

# Fungsi untuk memuat dan melakukan skalasi data
def load_and_scale_data(file_path, scaler=None):
    df = pd.read_csv(file_path, index_col='tgl_perawatan', parse_dates=True)
    df = df.resample('D').sum()  # Mengubah data menjadi data harian
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df)
    scaled_data = scaler.transform(df)
    return df, scaled_data, scaler

# Fungsi untuk melakukan prediksi
def predict_future(model, data, scaler, n_input, n_features, n_days):
    predictions = []

    # Gunakan batch terakhir dari data yang ada untuk memulai prediksi
    current_batch = data[-n_input:].reshape((1, n_input, n_features))

    for _ in range(n_days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Inversi transformasi skala
    true_predictions = scaler.inverse_transform(predictions)

    return true_predictions

# Fungsi utama untuk memuat model dan melakukan prediksi
def main(historical_data_path, model_path, n_input, n_features, n_months):
    # Muat model dengan custom_objects
    custom_objects = {'Orthogonal': keras.initializers.Orthogonal}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    # Muat dan skalakan data historis terbaru
    df, scaled_data, scaler = load_and_scale_data(historical_data_path)

    # Hitung jumlah hari untuk prediksi berdasarkan jumlah bulan
    n_days = int(n_months) * 30  # Asumsi 30 hari per bulan untuk kesederhanaan

    # Lakukan prediksi
    predictions = predict_future(model, scaled_data, scaler, n_input, n_features, n_days)

    # Buat DataFrame untuk hasil prediksi harian
    future_dates = pd.date_range(start=df.index[-1], periods=n_days + 1, freq='D')[1:]
    daily_prediction_df = pd.DataFrame(data=predictions, index=future_dates, columns=['Predictions'])

    # Agregasi hasil prediksi harian menjadi hasil prediksi bulanan
    monthly_prediction_df = daily_prediction_df.resample('M').sum()

    return monthly_prediction_df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    code = data.get('code')
    n_months = data.get('n_months')

    # Pastikan kode ada di dictionary
    if code in code_dict:
        model_path = code_dict[code]['model']
        file_path = code_dict[code]['file']
        n_input = 7
        n_features = 1

        monthly_prediction_df = main(file_path, model_path, n_input, n_features, n_months)
        
        # Convert the DataFrame to JSON
        predictions = monthly_prediction_df.to_dict()
        converted_predictions = {timestamp.strftime('%Y-%m-%d'): value['Predictions'] for timestamp, value in monthly_prediction_df.iterrows()}
        return jsonify(converted_predictions)

    else:
        return jsonify({"error": "Kode tidak ditemukan. Silakan coba lagi."}), 400

if __name__ == '__main__':
    app.run(debug=True)
