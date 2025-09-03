import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(page_title="Prediksi I/O Prefetching", layout="centered")

st.title("💡 Sistem Prediksi Akses I/O untuk Prefetching Cerdas")
st.markdown("""
Aplikasi ini mendemonstrasikan kemampuan model Machine Learning dalam memprediksi 
`file_offset` dan `request_io_size_bytes` dari operasi I/O 'READ' berikutnya.
Model ini dilatih untuk mengantisipasi pola akses data pada sistem operasi, 
membantu optimasi strategi *prefetching*.
""")

# --- 1. Muat Model dan Fitur yang Sudah Dilatih ---
#@st.cache_resource # Cache resource agar model hanya dimuat sekali
def load_artifacts():
    try:
        model = joblib.load('best_io_prefetch_model.joblib')
        num_features = joblib.load('numerical_features.joblib')
        cat_features = joblib.load('categorical_features.joblib')
        return model, num_features, cat_features
    except FileNotFoundError:
        st.error("Error: File model atau fitur tidak ditemukan. Pastikan Anda sudah menjalankan script pelatihan dan menyimpannya di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau fitur: {e}")
        st.stop()

model_pipeline, numerical_features, categorical_features = load_artifacts()

# --- 2. Fungsi untuk Membuat Fitur Input dari Data User ---
# Fungsi ini harus sama persis dengan logika feature engineering saat training
def create_features_for_prediction(current_io_data: pd.Series, last_read_t_minus_1_data: dict) -> pd.DataFrame:
    """
    Membuat DataFrame fitur yang diperlukan model untuk prediksi, berdasarkan I/O saat ini
    dan I/O READ sebelumnya (t-1).
    """
    input_features_for_prediction = pd.Series(dtype=object)

    # Isi fitur `last_` dengan data dari `last_read_t_minus_1_data` (I/O pada t-1)
    input_features_for_prediction['last_file_offset'] = last_read_t_minus_1_data['file_offset']
    input_features_for_prediction['last_request_io_size_bytes'] = last_read_t_minus_1_data['request_io_size_bytes']
    input_features_for_prediction['last_start_time'] = last_read_t_minus_1_data['start_time']
    input_features_for_prediction['last_op_type'] = 'READ' # Model hanya fokus READ

    # Hitung fitur delta menggunakan `current_io_data` (I/O pada t) dan `last_read_t_minus_1_data` (I/O pada t-1)
    input_features_for_prediction['offset_delta'] = current_io_data['file_offset'] - input_features_for_prediction['last_file_offset']
    input_features_for_prediction['size_delta'] = current_io_data['request_io_size_bytes'] - input_features_for_prediction['last_request_io_size_bytes']
    input_features_for_prediction['time_since_last_io'] = current_io_data['start_time'] - input_features_for_prediction['last_start_time']
    
    # Indikator sekuensial
    input_features_for_prediction['is_sequential_last_io'] = int(
        current_io_data['file_offset'] == (input_features_for_prediction['last_file_offset'] + input_features_for_prediction['last_request_io_size_bytes'])
    )
    
    # Buat DataFrame dari Series ini
    input_df = pd.DataFrame([input_features_for_prediction])
    
    # Pastikan urutan dan nama kolom sesuai dengan yang diharapkan oleh preprocessor model
    # Kolom 'last_op_type' tidak ada di `numerical_features` atau `categorical_features` yang disimpan
    # karena kita sudah filter 'READ' dan OneHotEncoder akan membuat kolom dari 'is_sequential_last_io'.
    # Jadi, kita tidak perlu memasukkannya secara eksplisit di sini.
    
    required_cols_order = numerical_features + categorical_features
    # Untuk OneHotEncoder, is_sequential_last_io akan menjadi is_sequential_last_io_0 dan is_sequential_last_io_1.
    # Namun, ColumnTransformer akan mengurusnya secara otomatis selama kolom input_df memiliki 'is_sequential_last_io'.
    
    # Periksa dan tambahkan kolom yang mungkin hilang jika OHE memiliki kategori yang tidak muncul di data training
    # Ini penting agar OneHotEncoder tidak error.
    # Karena kita hanya punya 1 kategori biner 'is_sequential_last_io', ini sederhana.
    
    return input_df[required_cols_order]

# --- 3. Antarmuka Pengguna Streamlit ---

st.header("Masukkan Data I/O Terbaru:")
st.markdown("Isi detail I/O READ yang **baru saja terjadi (t)** dan I/O READ **sebelumnya (t-1)** untuk memprediksi I/O berikutnya **(t+1)**.")

# Input untuk I/O (t-1)
st.subheader("I/O READ Sebelumnya (t-1):")
prev_offset_t_minus_1 = st.number_input("File Offset (t-1)", value=16384, min_value=0, step=1, key='prev_offset_t_minus_1')
prev_size_t_minus_1 = st.number_input("Request I/O Size (t-1)", value=4096, min_value=1, step=1, key='prev_size_t_minus_1')
prev_time_t_minus_1 = st.number_input("Start Time (t-1)", value=10.7, min_value=0.0, step=0.1, key='prev_time_t_minus_1')

# Input untuk I/O (t) (Current)
st.subheader("I/O READ Saat Ini (t):")
current_offset_t = st.number_input("File Offset (t)", value=20480, min_value=0, step=1, key='current_offset_t')
if current_offset_t < (prev_offset_t_minus_1 + prev_size_t_minus_1):
    st.warning("Perhatian: 'File Offset (t)' harus lebih besar atau sama dengan 'File Offset (t-1)' + 'Request I/O Size (t-1)' untuk memastikan urutan yang benar.")
    st.warning(f"Nilai minimum yang valid adalah {prev_offset_t_minus_1 + prev_size_t_minus_1}.")
else:
    current_size_t = st.number_input("Request I/O Size (t)", value=4096, min_value=1, step=1, key='current_size_t')
    current_time_t = st.number_input("Start Time (t)", value=10.9, min_value=0.0, step=0.1, key='current_time_t')
    predict_button = st.button("Prediksi I/O Berikutnya (t+1)")

    if predict_button:
        if current_time_t <= prev_time_t_minus_1:
            st.warning("Perhatian: 'Start Time (t)' harus lebih besar dari 'Start Time (t-1)'.")
        else:
            # Siapkan data untuk fungsi create_features_for_prediction
            current_io_data_series = pd.Series({
                'file_offset': current_offset_t,
                'request_io_size_bytes': current_size_t,
                'start_time': current_time_t,
                'op_type': 'READ' # Diset permanen karena model hanya untuk READ
            })

            last_read_t_minus_1_data_dict = {
                'file_offset': prev_offset_t_minus_1,
                'request_io_size_bytes': prev_size_t_minus_1,
                'start_time': prev_time_t_minus_1
            }
            
            # Buat fitur
            try:
                input_df_for_pred = create_features_for_prediction(
                    current_io_data_series,
                    last_read_t_minus_1_data_dict
                )

                # Lakukan prediksi
                prediction = model_pipeline.predict(input_df_for_pred)
                predicted_offset = prediction[0, 0]
                predicted_size = prediction[0, 1]

                st.success("Prediksi Berhasil!")
                st.subheader("Hasil Prediksi untuk I/O Berikutnya (t+1):")
                st.write(f"**Predicted Next File Offset:** `{predicted_offset:,.2f}` bytes")
                st.write(f"**Predicted Next Request I/O Size:** `{predicted_size:,.2f}` bytes")
                st.markdown("""
                Prediksi ini dapat digunakan oleh sistem prefetching untuk mengambil data
                yang relevan ke *cache* lebih awal, sehingga mengurangi latensi.
                """)
            except ValueError as ve:
                st.error(f"Error dalam pembuatan fitur: {ve}. Pastikan input numerik valid.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memprediksi: {e}")

    st.markdown("---")
    st.markdown("Dibuat oleh Associate Data Scientist | OSync Innovations")