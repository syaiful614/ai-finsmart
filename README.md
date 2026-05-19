# FinSmart — AI Service

**Coding Camp 2026 powered by DBS Foundation**
**CC26-PSU407 | Tema: Revolusi Teknologi Keuangan (Fintech) untuk Generasi Muda**

AI Engineer: Muhammad Syaiful

---

## Deskripsi Proyek

FinSmart adalah aplikasi pengelolaan keuangan pribadi berbasis AI untuk generasi muda. Repository ini berisi seluruh komponen AI yang dibangun oleh AI Engineer, meliputi model klasifikasi pengeluaran, prediksi pengeluaran bulanan (LSTM), chatbot keuangan (FinBot), sistem rekomendasi kesiapan investasi, dan klasifikasi perilaku keuangan.

---

## Checklist Capstone

### Main Quest
| No | Komponen | Status |
|----|----------|--------|
| 1 | Model Deep Learning TensorFlow Functional API | ✅ Selesai |
| 2 | Custom Callback (FinSmartCallback & FinSmartLSTMCallback) | ✅ Selesai |
| 3 | Simpan model format `.keras` | ✅ Selesai |
| 4 | Kode Inference (FinSmartInference & FinSmartLSTMInference) | ✅ Selesai |

### Side Quest
| No | Komponen | Status |
|----|----------|--------|
| 1 | REST API FastAPI — deploy di Railway | ✅ Selesai |
| 2 | Akurasi >= 85% (XGBoost 99.83%) | ✅ Selesai |
| 3 | Model LSTM Spending Prediction (R2 = 0.9689, MAPE = 6.37%) | ✅ Selesai |
| 4 | Integrasi model DS (financial_type_classifier.pkl) | ✅ Selesai |

---

## Arsitektur Pipeline AI

```
Input Transaksi (Front-End)
        |
        v
Back-End Node.js
        |
        v
AI Service FastAPI (Repository ini) — v3.0.0
        |
        |--- POST /classify         --> XGBoost (Klasifikasi Kategori)
        |--- POST /behavior         --> Decision Tree (Perilaku Keuangan)
        |--- POST /predict-spending --> LSTM TensorFlow (Prediksi Spending)
        |--- POST /rekomendasi      --> Rule-Based BLR + SR (Kesiapan Investasi)
        |--- POST /finbot/chat      --> Hugging Face BART (Chatbot)
        |
        v
Response JSON ke Front-End
```

---

## Model yang Digunakan

### 1. Klasifikasi Kategori Pengeluaran
- **Model utama**: XGBoost (`xgb_model.pkl`)
- **Model pendukung**: TensorFlow Functional API (`finsmart_model.keras`)
- **Input**: Amount, Payment_Method, Week_Day, Month, Time_Of_Day, MerchantName, Day
- **Output**: 9 kategori pengeluaran
- **Akurasi XGBoost**: 99.83%
- **Custom Callback**: FinSmartCallback

### 2. Klasifikasi Perilaku Keuangan
- **Model**: Decision Tree (`financial_type_classifier.pkl`) dari tim Data Scientist
- **Input**: Income, Needs, Wants, Savings
- **Output**: hemat / normal / boros
- **Acuan**: Prinsip 50/30/20

### 3. Prediksi Pengeluaran Bulanan (LSTM)
- **Model**: LSTM 2 layer TensorFlow Functional API (`finsmart_lstm.keras`)
- **Input**: Histori 3 bulan (income, spending, savings, per kategori)
- **Output**: Prediksi total pengeluaran bulan depan
- **Metrik**: MAE Rp 454.250 | RMSE Rp 638.530 | MAPE 6.37% | R2 0.9689
- **Custom Callback**: FinSmartLSTMCallback

### 4. FinBot — Chatbot Keuangan
- **Model**: Hugging Face `facebook/bart-large-mnli`
- **Teknik**: Zero-Shot Classification
- **Input**: Pertanyaan keuangan bahasa Indonesia
- **Output**: Respons edukasi keuangan (8 intent)

### 5. Rekomendasi Kesiapan Investasi
- **Metode**: Rule-Based (BLR + SR)
- **BLR** = Tabungan Total / Total Pengeluaran Bulanan
- **SR** = Tabungan Bulanan / Income Bulanan x 100%
- **Output**: Not Ready / Moderately Ready / Ready

---

## Kategori Pengeluaran (9 Kelas)

| No | Kategori |
|----|----------|
| 1 | Food & Dining |
| 2 | Transportation |
| 3 | Shopping |
| 4 | Groceries |
| 5 | Bills & Utilities |
| 6 | Entertainment |
| 7 | Health |
| 8 | Education |
| 9 | Others |

---

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Health check |
| POST | `/classify` | Klasifikasi kategori pengeluaran |
| POST | `/behavior` | Klasifikasi perilaku keuangan |
| POST | `/predict-spending` | Prediksi pengeluaran bulan depan (LSTM) |
| POST | `/rekomendasi` | Analisis kesiapan investasi |
| POST | `/finbot/chat` | Chatbot keuangan FinBot |
| GET | `/model-info` | Informasi model |
| GET | `/valid-values` | Nilai valid tiap fitur |
| GET | `/docs` | Swagger UI |

---

## Struktur File

```
ai-finsmart/
|
|-- FinSmart_AI_Engineer_UPDATED_v2.ipynb          # Notebook klasifikasi transaksi
|-- FinSmart_FinBot_Rekomendasi_UPDATED_v2.ipynb   # Notebook FinBot & Rekomendasi
|-- FinSmart_SpendingPrediction_LSTM.ipynb         # Notebook LSTM Spending Prediction
|
|-- main.py                                        # FastAPI basic
|-- main_complete.py                               # FastAPI lengkap (v3.0.0)
|-- Procfile                                       # Konfigurasi Railway
|-- requirements.txt                               # Library dependencies
|
|-- finsmart_model.keras                           # Model TF Functional API (klasifikasi)
|-- finsmart_lstm.keras                            # Model LSTM (spending prediction)
|-- xgb_model.pkl                                  # Model XGBoost final
|-- financial_type_classifier.pkl                  # Model behavior (dari DS)
|-- encoders.pkl                                   # Label encoders
|-- scaler.pkl                                     # StandardScaler (klasifikasi)
|-- scaler_lstm.pkl                                # MinMaxScaler (LSTM)
|
|-- model_metadata.json                            # Metadata & performa model
|-- model_metadata_lstm.json                       # Metadata model LSTM
|-- training_log.json                              # Log training klasifikasi
|-- training_log_lstm.json                         # Log training LSTM
|-- arsitektur_ai.json                             # Dokumentasi arsitektur
|-- finbot_chat_history.json                       # Bukti FinBot berjalan
|-- contoh_rekomendasi.json                        # Bukti rekomendasi berjalan
|
|-- training_history.png                           # Grafik training klasifikasi
|-- confusion_matrix.png                           # Confusion matrix XGBoost
|-- lstm_training_history.png                      # Grafik training LSTM
|-- lstm_prediction_result.png                     # Grafik prediksi LSTM
```

---

## Cara Menjalankan API Lokal

```bash
pip install -r requirements.txt
uvicorn main_complete:app --host 0.0.0.0 --port 8000 --reload
```

Akses Swagger UI: `http://localhost:8000/docs`

---

## Contoh Request

### POST /classify
```json
{
  "Amount": 250000,
  "Payment_Method": "Credit Card",
  "Week_Day": "Saturday",
  "Month": "May",
  "Time_Of_Day": "Evening",
  "MerchantName": "Amazon",
  "Day": 10
}
```

### POST /behavior
```json
{
  "Income": 5000000,
  "Needs": 2500000,
  "Wants": 1000000,
  "Savings": 1500000
}
```

### POST /predict-spending
```json
{
  "histori": [
    {
      "income": 5000000, "total_spending": 3200000, "savings": 1800000,
      "food_dining": 800000, "transportation": 400000, "shopping": 500000,
      "groceries": 300000, "bills_utilities": 400000, "entertainment": 200000,
      "health": 100000, "education": 200000, "others": 300000
    },
    {
      "income": 5000000, "total_spending": 3500000, "savings": 1500000,
      "food_dining": 900000, "transportation": 450000, "shopping": 600000,
      "groceries": 350000, "bills_utilities": 400000, "entertainment": 250000,
      "health": 150000, "education": 200000, "others": 200000
    },
    {
      "income": 5000000, "total_spending": 3800000, "savings": 1200000,
      "food_dining": 1000000, "transportation": 500000, "shopping": 700000,
      "groceries": 400000, "bills_utilities": 400000, "entertainment": 300000,
      "health": 100000, "education": 200000, "others": 200000
    }
  ]
}
```

### POST /rekomendasi
```json
{
  "tabungan_total": 15000000,
  "total_pengeluaran_bulanan": 3500000,
  "tabungan_bulanan": 1000000,
  "income_bulanan": 5000000
}
```

### POST /finbot/chat
```json
{
  "pertanyaan": "Bagaimana cara menabung lebih banyak setiap bulan?"
}
```

---

## Tim CC26-PSU407

| Nama | Peran |
|------|-------|
| Zahwa Putri Vanisa | Data Scientist |
| A. Rafly Sahrul | Full-Stack Web Developer (Back-End) |
| Ahmad Anta Wirangga | Full-Stack Web Developer (Front-End) |
| Fahira Zahra Fitria Rahma | Data Scientist |
| Muhammad Syaiful | AI Engineer |

---

## Repository & Link Terkait

| Nama | Link |
|------|------|
| AI Service (repo ini) | https://github.com/FinSmartTeam/ai-finsmart |
| Back-End | https://github.com/FinSmartTeam/backend-FinSmart |
| Front-End | https://github.com/FinSmartTeam/finsmart-frontend |
| AI API Live | https://web-production-6ba83.up.railway.app |
| Back-End URL | https://backend-fin-smart.vercel.app |
| Swagger UI | https://web-production-6ba83.up.railway.app/docs |
