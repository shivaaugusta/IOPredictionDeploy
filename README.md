# Aplikasi Prediksi I/O Trace Menggunakan Machine Learning

Aplikasi ini menggunakan model machine learning untuk memprediksi I/O trace berdasarkan data historis. Model yang digunakan adalah Random Forest Classifier yang telah dilatih dan disimpan dalam file `best_io_prefetch_model.joblib`. Paper rujukan dataset dalam proyek ini adalah [Thesios: Synthesizing Accurate Counterfactual I/O Traces from I/O Samples](https://dl.acm.org/doi/10.1145/3620666.3651337).

## Instalasi

1. Clone repository ini:
   ```bash
    git clone https://github.com/shivaaugusta/I-O_Trace_Prediction.git
    cd I-O_Trace_Prediction
   ```
2. Install dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
3. Download file model yang telah dilatih:
   - [best_io_prefetch_model.joblib](https://drive.google.com/file/d/1R4Ni2bhBkBr7Ctw6FU3Bap7TsUw_IsWW/view?usp=sharing)
   - [categorical_features.joblib](https://drive.google.com/file/d/1LVfaCl6T3c1h08wEJifmlYLlemn_nKvA/view?usp=sharing)
   - [numerical_features.joblib](https://drive.google.com/file/d/1MGq_DZbMJSRyjEXLRyLG1Ii1SN3XAjbw/view?usp=sharing)
4. Letakkan file-file tersebut di direktori utama proyek.

## Penggunaan

1. Pastikan Anda berada di direktori utama proyek.
2. Jalankan aplikasi dengan perintah berikut:
   ```bash
   streamlit run app.py
   ```
3. Ikuti instruksi dalam memasukkan data I/O trace yang ingin diprediksi.
