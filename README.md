# Smart Predictive Maintenance Simulator

Proyek UAS Pemodelan dan Simulasi — Minggu 16 (Integrasi Akhir).
Simulator memprediksi **risiko kegagalan mesin** menggunakan Machine Learning, menjelaskan
hasil prediksi dengan **SHAP (XAI)**, dan memberikan **rekomendasi prioritas perawatan**
menggunakan metode **SAW (SPK)** dengan bobot kriteria dari **AHP**.

## Alur Sistem (Pipeline)

```
Slider (Streamlit UI)
      │
      ▼
StandardScaler (models/scaler.joblib)
      │
      ▼
RandomForestRegressor (models/model.joblib)  ──► Skor Risiko
      │                                              │
      ▼                                              ▼
SHAP Explainer (XAI)                    Matriks Keputusan SPK
      │                                  (Risiko + Biaya + Efisiensi)
      ▼                                              │
Penjelasan kontribusi fitur                          ▼
                                          SAW + Bobot AHP ──► Ranking Prioritas Perawatan
```

## Struktur Folder

```
Predictive_Maintenance/
├── app.py                 # Dashboard utama Streamlit
├── train_model.py         # Script training model ML
├── requirements.txt
├── README.md
├── models/
│   ├── model.joblib        # RandomForestRegressor terlatih
│   ├── scaler.joblib        # StandardScaler
│   └── model_metadata.json  # RMSE/MAE/R2 + rentang valid input
├── data/
│   ├── machine_data.csv     # Dataset sintetis
│   └── generate_data.py     # Script pembuat dataset
└── utils/
    ├── saw.py            # Implementasi SAW + AHP
    ├── shap_utils.py     # Helper SHAP (waterfall, summary, narasi)
    └── helper.py          # Validasi input, integrasi matriks SPK, anonimisasi, dll.
```

## Cara Menjalankan

1. **Buat virtual environment** (disarankan, untuk Replayability):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies** (versi sudah dikunci di `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```

3. **(Opsional) Generate ulang dataset:**
   ```bash
   python data/generate_data.py
   ```

4. **Latih model** (menghasilkan file di folder `models/`):
   ```bash
   python train_model.py
   ```

5. **Jalankan dashboard:**
   ```bash
   streamlit run app.py
   ```

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| Slider What-If | Ubah Suhu, Vibrasi, Tekanan, Usia Mesin, Jam Operasi, Beban secara real-time |
| Prediksi Risiko | RandomForestRegressor memprediksi skor risiko 0–100 |
| Baseline vs Current | Simpan satu skenario sebagai baseline, bandingkan dengan skenario aktif |
| SHAP Waterfall | Penjelasan kontribusi fitur untuk satu prediksi spesifik |
| Tabel Kontribusi Fitur | Narasi otomatis "fitur X menaikkan/menurunkan risiko sebesar Y" |
| SAW + AHP | Ranking prioritas perawatan antar mesin berdasarkan Risiko, Biaya, Efisiensi |
| Validasi Input | Peringatan otomatis jika input di luar rentang data training (potensi drift) |
| Download CSV | Unduh hasil ranking SAW |

## Catatan Etika & Privasi

Dataset bersifat sintetis dan tidak berisi data identitas (PII). Jika model dikembangkan
dengan data produksi nyata, kolom seperti ID operator atau lokasi presisi harus dihapus
sebelum demo publik — gunakan `utils.helper.anonymize_dataframe()`.

## Replayability Checklist

- [x] `requirements.txt` mencantumkan versi spesifik setiap library
- [x] Path model menggunakan `os.path.join` relatif terhadap lokasi file (bukan hardcode absolut)
- [x] Dataset dan proses training dapat dijalankan ulang dari nol (`generate_data.py` + `train_model.py`)
