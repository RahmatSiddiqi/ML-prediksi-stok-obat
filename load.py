import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

# Dictionary untuk memetakan kode obat ke path model dan file data
code_dict = {
    'Lesipar': {'model': r'model\lesipar.h5', 'file': r'file\lesipar.csv'},
    'Metronidazol': {'model': r'model\metronidazol.h5', 'file': r'file\metronidazol.csv'},
    'Alprazolam': {'model': r'model\alprazolam.h5', 'file': r'file\alprazolam.csv'},
    'Lansoprazol': {'model': r'model\lansoprazol.h5', 'file': r'file\lansoprazol.csv'},
    'Digoksin': {'model': r'model\digoksin.h5', 'file': r'file\digoksin.csv'},
    'Merlopam': {'model': r'model\merlopam.h5', 'file': r'file\merlopam.csv'},
    'Nifedifin': {'model': r'model\nifedifin.h5', 'file': r'file\nifedifin.csv'},
    'Ramipril25': {'model': r'model\ramipril2,5.h5', 'file': r'file\ramipril2,5.csv'},
    'Amitriptilin': {'model': r'model\amitriptilin.h5', 'file': r'file\amitriptilin.csv'},
    'Arkine': {'model': r'model\arkine.h5', 'file': r'file\arkine.csv'},
    'Kodein10mg': {'model': r'model\kodein10mg.h5', 'file': r'file\kodein10mg.csv'},
    'Alprazolam05': {'model': r'model\alprazolam05.h5', 'file': r'file\alprazolam05.csv'},
    'Sefiksim100': {'model': r'model\sefiksim100.h5', 'file': r'file\sefiksim100.csv'},
    'Parasetamol': {'model': r'model\parasetamol.h5', 'file': r'file\parasetamol.csv'},
    'Ramipril': {'model': r'model\ramipril.h5', 'file': r'file\ramipril.csv'},
    'Sefiksim': {'model': r'model\sefiksim.h5', 'file': r'file\sefiksim.csv'},
    'Ranitidin': {'model': r'model\ranitidin.h5', 'file': r'file\ranitidin.csv'},
    'Natriumbikarbonat': {'model': r'model\natriumbikarbonat.h5', 'file': r'file\natriumbikarbonat.csv'},
    'Misoprostol': {'model': r'model\misoprostol.h5', 'file': r'file\misoprostol.csv'},
    'Setirizin': {'model': r'model\setirizin.h5', 'file': r'file\setirizin.csv'},
    'Siprofloksasin': {'model': r'model\siprofloksasin.h5', 'file': r'file\siprofloksasin.csv'},
    'Klopidogrel': {'model': r'model\klopidogrel.h5', 'file': r'file\klopidogrel.csv'},
    'Isosorbid': {'model': r'model\isosorbid.h5', 'file': r'file\isosorbid.csv'},
    'Metformin': {'model': r'model\metformin.h5', 'file': r'file\metformin.csv'},
    'Metilprednisolon': {'model': r'model\metilprednisolon.h5', 'file': r'file\metilprednisolon.csv'},
    'Natriumdiklofenak50': {'model': r'model\natriumdiklofenak50.h5', 'file': r'file\natriumdiklofenak50.csv'},
    'Stimox': {'model': r'model\stimox.h5', 'file': r'file\stimox.csv'},
    'Furosemid': {'model': r'model\furosemid.h5', 'file': r'file\furosemid.csv'},
    'Gabapentin': {'model': r'model\gabapentin.h5', 'file': r'file\gabapentin.csv'},
    'Neurohax': {'model': r'model\neurohax.h5', 'file': r'file\neurohax.csv'},
    # Tambahkan lebih banyak obat sesuai kebutuhan
}

actual_data_paths = {
    'Lesipar': 'filefullcsv/lesipar.csv',
    'Metronidazol': 'filefullcsv/metronidazol.csv',
    'Alprazolam': 'filefullcsv/alprazolam.csv',
    'Lansoprazol': 'filefullcsv/lansoprazol.csv',
    'Digoksin': 'filefullcsv/digoksin.csv',
    'Merlopam': 'filefullcsv/merlopam.csv',
    'Nifedifin': 'filefullcsv/nifedifin.csv',
    'Ramipril25': 'filefullcsv/ramipril2,5.csv',
    'Amitriptilin': 'filefullcsv/amitriptilin.csv',
    'Arkine': 'filefullcsv/arkine.csv',
    'Kodein10mg': 'filefullcsv/kodein10mg.csv',
    'Alprazolam05': 'filefullcsv/alprazolam05.csv',
    'Sefiksim100': 'filefullcsv/sefiksim100.csv',
    'Parasetamol': 'filefullcsv/parasetamol.csv',
    'Ramipril': 'filefullcsv/ramipril.csv',
    'Sefiksim': 'filefullcsv/sefiksim.csv',
    'Ranitidin': 'filefullcsv/ranitidin.csv',
    'Natriumbikarbonat': 'filefullcsv/natriumbikarbonat.csv',
    'Misoprostol': 'filefullcsv/misoprostol.csv',
    'Setirizin': 'filefullcsv/setirizin.csv',
    'Siprofloksasin': 'filefullcsv/siprofloksasin.csv',
    'Klopidogrel': 'filefullcsv/klopidogrel.csv',
    'Isosorbid': 'filefullcsv/isosorbid.csv',
    'Metformin': 'filefullcsv/metformin.csv',
    'Metilprednisolon': 'filefullcsv/metilprednisolon.csv',
    'Natriumdiklofenak50': 'filefullcsv/natriumdiklofenak50.csv',
    'Stimox': 'filefullcsv/stimox.csv',
    'Furosemid': 'filefullcsv/furosemid.csv',
    'Gabapentin': 'filefullcsv/gabapentin.csv',
    'Neurohax': 'filefullcsv/neurohax.csv',
    # Sesuaikan dengan nama file dan path yang sesuai
}

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow import keras
# import matplotlib.pyplot as plt

# Fungsi untuk memuat dan melakukan skalasi data
def load_and_scale_data(file_path, scaler=None):
    df = pd.read_csv(file_path, index_col='tgl_perawatan', parse_dates=True)
    df = df.resample('D').sum()  # Ubah data menjadi data harian
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df)
    scaled_data = scaler.transform(df)
    return df, scaled_data, scaler

# Fungsi untuk melakukan prediksi menggunakan model yang diload
def predict_future(model, data, scaler, n_input, n_days):
    predictions = []

    # Gunakan batch terakhir dari data yang ada untuk memulai prediksi
    current_batch = data[-n_input:].reshape((1, n_input, 1))

    for _ in range(n_days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Inversi transformasi skala
    true_predictions = scaler.inverse_transform(predictions)

    return true_predictions

# Fungsi utama untuk memuat model dan melakukan prediksi
def load_and_predict(file_path, model_path, n_input, n_months):
    # Load model
    model = keras.models.load_model(model_path)

    # Load data dan skalakan data historis
    df, scaled_data, scaler = load_and_scale_data(file_path)

    # Ambil jumlah hari data historis
    historical_days = len(df)

    # Hitung jumlah hari untuk prediksi berdasarkan jumlah bulan
    n_days = int(n_months) * 30  # Asumsi 30 hari per bulan

    # Lakukan prediksi
    predictions = predict_future(model, scaled_data, scaler, n_input, n_days)

    # Buat DataFrame untuk hasil prediksi harian
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')
    daily_prediction_df = pd.DataFrame(data=predictions, index=future_dates, columns=['Predictions'])

    # Agregasi hasil prediksi harian menjadi hasil prediksi bulanan
    monthly_prediction_df = daily_prediction_df.resample('M').sum()

    return monthly_prediction_df

# Fungsi untuk memuat data aktual dari file CSV
def load_actual_data(file_path):
    df = pd.read_csv(file_path, index_col='tgl_perawatan', parse_dates=True)
    df = df.resample('M').sum()  # Resample data harian menjadi data bulanan
    return df

# Fungsi untuk mencetak hasil prediksi
def print_predictions(prediction_result):
    print("\nHasil Prediksi Bulanan:")
    for index, row in prediction_result.iterrows():
        print(f"{index.strftime('%Y-%m-%d')}: {row['Predictions']}")

# Fungsi untuk mencetak data aktual dari bulan Agustus sampai Desember 2022
def print_actual_data(actual_data):
    print("\nData Aktual Bulanan (Agustus-Desember 2022):")
    for index, row in actual_data.iterrows():
        print(f"{index.strftime('%Y-%m-%d')}: {row['total_jumlah_per_hari']}")

# Fungsi untuk menghitung MAPE (Mean Absolute Percentage Error) bulanan
def calculate_mape(actual_data, prediction_result):
    mape_list = []
    for index in actual_data.index:
        actual_value = actual_data.loc[index, 'total_jumlah_per_hari']
        predicted_value = prediction_result.loc[index, 'Predictions']
        if not pd.isnull(actual_value) and not pd.isnull(predicted_value):
            error = np.abs((actual_value - predicted_value) / actual_value)
            mape_list.append(error * 100)
    mape = np.mean(mape_list)
    return mape, mape_list

# Fungsi untuk memproses prediksi dan evaluasi untuk satu kode obat
def process_predictions(code, n_months, n_input=7):
    if code in code_dict:
        model_path = code_dict[code]['model']
        file_path = code_dict[code]['file']
        actual_data_path = actual_data_paths.get(code)  # Menggunakan actual_data_paths

        if not actual_data_path:
            print(f"Path data aktual untuk {code} tidak ditemukan.")
            return

        # Melakukan prediksi
        prediction_result = load_and_predict(file_path, model_path, n_input, n_months)

        # Memuat data aktual dan mengambil data dari Agustus 2022 hingga Desember 2022
        actual_data = load_actual_data(actual_data_path)
        actual_data_aug_to_dec_2022 = actual_data.loc['2022-08-01':'2022-12-31']

        # Plotting perbandingan aktual vs prediksi
        plt.figure(figsize=(12, 6))
        plt.plot(actual_data_aug_to_dec_2022.index, actual_data_aug_to_dec_2022['total_jumlah_per_hari'], label='Aktual', marker='o')
        plt.plot(prediction_result.index, prediction_result['Predictions'], label='Prediksi', marker='x')
        plt.title('Perbandingan Aktual vs Prediksi Pengeluaran Bulanan (Agustus-Desember 2022) - {}'.format(code))
        plt.xlabel('Tanggal')
        plt.ylabel('Jumlah')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Cetak hasil prediksi
        print_predictions(prediction_result)

        # Cetak data aktual
        print_actual_data(actual_data_aug_to_dec_2022)

        # Hitung MAPE bulanan
        mape, mape_list = calculate_mape(actual_data_aug_to_dec_2022, prediction_result)
        print("\nMAPE Bulanan:")
        for i, month in enumerate(actual_data_aug_to_dec_2022.index.month_name()):
            print(f"{month} {actual_data_aug_to_dec_2022.index.year[i]}: {mape_list[i]:.2f}%")
        print(f"Rata-rata MAPE: {mape:.2f}%")
    else:
        print(f"Kode obat {code} tidak ditemukan dalam dictionary.")

# Fungsi untuk input dari terminal
def user_input():
    code = input("Masukkan kode obat: ").strip()
    n_months = int(input("Masukkan jumlah bulan untuk diprediksi: ").strip())
    return code, n_months

# Contoh penggunaan untuk banyak obat
if __name__ == '__main__':
    code, n_months = user_input()
    process_predictions(code, n_months)
