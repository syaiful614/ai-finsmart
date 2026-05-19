"""
FinSmart AI Service — main_complete_v2.py
=========================================
CC26-PSU407 | Muhammad Syaiful | AI Engineer

PERUBAHAN dari v1 ke v2:
- /classify        : Diupdate sesuai dataset baru DS (7 fitur, encoders baru)
- /behavior        : BARU — pakai model financial_type_classifier.pkl dari DS
- /rekomendasi     : Diupdate rumus BLR + SR sesuai dokumen resmi DS
- /finbot          : Tidak berubah

PERUBAHAN dari v2 ke v3:
- /predict-spending: BARU — prediksi pengeluaran bulan depan (LSTM)
"""

import numpy as np
import pickle
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    from transformers import pipeline as hf_pipeline
    FINBOT_AVAILABLE = True
except ImportError:
    FINBOT_AVAILABLE = False

# ──────────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────────
app = FastAPI(
    title="FinSmart AI Service",
    description="API lengkap: klasifikasi pengeluaran, behavior keuangan, FinBot, dan rekomendasi investasi",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ──────────────────────────────────────────────
# LOAD MODELS & ARTIFACTS
# ──────────────────────────────────────────────

# Model klasifikasi transaksi (XGBoost — dataset baru)
xgb_model    = joblib.load("xgb_model.pkl")
encoders     = pickle.load(open("encoders.pkl", "rb"))
scaler       = pickle.load(open("scaler.pkl",   "rb"))
metadata     = json.load(open("model_metadata.json"))

# Model behavior keuangan (dari tim DS)
try:
    behavior_model = pickle.load(open("financial_type_classifier.pkl", "rb"))
except Exception as e:
    print(f"Warning: behavior_model gagal diload: {e}")
    behavior_model = None

# Model LSTM Spending Prediction
try:
    from tensorflow import keras as _keras
    lstm_model   = _keras.models.load_model("finsmart_lstm.keras")
    scaler_lstm  = pickle.load(open("scaler_lstm.pkl", "rb"))
    LSTM_AVAILABLE = True
    print("LSTM model berhasil diload")
except Exception as e:
    lstm_model   = None
    scaler_lstm  = None
    LSTM_AVAILABLE = False
    print(f"Warning: LSTM model gagal diload: {e}")

# FinBot — Hugging Face
finbot_clf = None
if FINBOT_AVAILABLE:
    try:
        finbot_clf = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    except Exception:
        finbot_clf = None

# ──────────────────────────────────────────────
# KONSTANTA FITUR — DATASET BARU (7 fitur)
# ──────────────────────────────────────────────
FITUR_ORDER = [
    "Amount",           # numerik, sudah di-scale
    "Payment_Method",   # encoded: 0=Credit Card, 1=Debit Card, 2=Net Banking, 3=UPI
    "Week_Day",         # encoded: 0=Friday, 1=Monday, 2=Saturday, 3=Sunday, 4=Thursday, 5=Tuesday, 6=Wednesday
    "Month",            # encoded: 0=April, 1=August, 2=December, 3=February, 4=January, 5=July, 6=June, 7=March, 8=May, 9=November, 10=October, 11=September
    "Time_Of_Day",      # encoded: 0=Afternoon, 1=Evening, 2=Morning, 3=Night
    "MerchantName",     # encoded: 0–107 (108 merchant)
    "Day",              # int: 1–31
]

PAYMENT_METHOD_MAP  = {"Credit Card": 0, "Debit Card": 1, "Net Banking": 2, "UPI": 3}
WEEK_DAY_MAP        = {"Friday": 0, "Monday": 1, "Saturday": 2, "Sunday": 3, "Thursday": 4, "Tuesday": 5, "Wednesday": 6}
TIME_OF_DAY_MAP     = {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3}
MONTH_MAP           = {"April": 0, "August": 1, "December": 2, "February": 3, "January": 4,
                       "July": 5, "June": 6, "March": 7, "May": 8, "November": 9, "October": 10, "September": 11}
MERCHANT_MAP        = {
    "1mg": 0, "A2B": 1, "ACT Fibernet": 2, "Adidas": 3, "Aha": 4,
    "Air India": 5, "AirAsia": 6, "Airtel": 7, "Ajio": 8, "Allen Solly": 9,
    "Amazon": 10, "Amazon Prime": 11, "Apollo Hospital": 12, "Apollo Pharmacy": 13,
    "Apple Music": 14, "Apple Store": 15, "Aster": 16, "Asus Store": 17,
    "BMTC": 18, "BSNL": 19, "Barbeque Nation": 20, "Biba": 21, "BigBazaar": 22,
    "Boat Lifestyle": 23, "Booking.com": 24, "Burger King": 25, "CCD": 26,
    "Chai Point": 27, "CloudNine": 28, "Cream Stone": 29, "Croma": 30,
    "DMart": 31, "Dell Exclusive": 32, "Dominos": 33, "Flipkart": 34,
    "Fortis": 35, "Gaana": 36, "Gas Agency": 37, "Goibibo": 38, "H&M": 39,
    "HP World": 40, "Hotstar": 41, "IRCTC": 42, "IndiGo": 43, "Jio Fiber": 44,
    "JioMart": 45, "KFC": 46, "Kauvery": 47, "Lenskart": 48, "Levis": 49,
    "MakeMyTrip": 50, "Max Fashion": 51, "McDonalds": 52, "MedPlus": 53,
    "Meesho": 54, "Meru Cabs": 55, "Metro Rail": 56, "Mi Store": 57,
    "More Supermarket": 58, "Myntra": 59, "Nature's Basket": 60, "Netflix": 61,
    "Netmeds": 62, "Nike": 63, "Nilgiris": 64, "Noise": 65, "Nykaa": 66,
    "OYO": 67, "Ola": 68, "OnePlus": 69, "Pantaloons": 70, "Paradise Biryani": 71,
    "Pepperfry": 72, "PharmEasy": 73, "Pizza Hut": 74, "Property Tax Online": 75,
    "Puma": 76, "Rapido": 77, "RedBus": 78, "Reliance Digital": 79,
    "Reliance Fresh": 80, "Samsung Store": 81, "Saravana Bhavan": 82,
    "Snapdeal": 83, "Sony Center": 84, "SonyLiv": 85, "Spencers": 86,
    "Spotify": 87, "Star Bazaar": 88, "Starbucks": 89, "Subway": 90,
    "TNEB": 91, "TSRTC": 92, "Taco Bell": 93, "Tata Cliq": 94, "US Polo": 95,
    "Uber": 96, "Vistara": 97, "Vivo Store": 98, "Vodafone Idea": 99,
    "Water Board": 100, "Woodland": 101, "Wrogn": 102, "Yatra": 103,
    "YouTube Premium": 104, "Yulu Bikes": 105, "Zara": 106, "Zee5": 107
}

CATEGORY_CLASSES = ["Bills", "Clothing", "Electronics", "Entertainment",
                    "Food", "Grocery", "Healthcare", "Online Shopping", "Transport", "Travel"]


# ──────────────────────────────────────────────
# KONSTANTA LSTM
# ──────────────────────────────────────────────
LSTM_FEATURES = [
    "income", "total_spending", "savings",
    "Food & Dining", "Transportation", "Shopping",
    "Groceries", "Bills & Utilities", "Entertainment",
    "Health", "Education", "Others"
]
LSTM_SEQUENCE_LENGTH = 3
# ──────────────────────────────────────────────
# FINBOT CONSTANTS
# ──────────────────────────────────────────────
INTENT_LABELS = [
    "tips menabung", "cara mengatur anggaran", "kategori pengeluaran",
    "investasi untuk pemula", "cara mengurangi pengeluaran",
    "target tabungan", "rekomendasi keuangan", "pengertian literasi keuangan",
]

RESPONSE_DB = {
    "tips menabung": (
        "Tips Menabung yang Efektif:\n"
        "1. Terapkan metode 50/30/20 — 50% kebutuhan, 30% keinginan, 20% tabungan.\n"
        "2. Simpan di awal bulan sebelum digunakan (pay yourself first).\n"
        "3. Buat rekening tabungan terpisah dari rekening harian.\n"
        "4. Mulai dari nominal kecil tapi konsisten — Rp 50.000/hari = Rp 1,5 juta/bulan.\n"
        "5. Gunakan fitur Target Tabungan di FinSmart untuk pantau progres kamu."
    ),
    "cara mengatur anggaran": (
        "Cara Mengatur Anggaran Bulanan:\n"
        "1. Catat semua pemasukan bulanan terlebih dahulu.\n"
        "2. Bagi pengeluaran ke kategori: Makanan, Transport, Tagihan, Hiburan, dll.\n"
        "3. Tentukan batas maksimal untuk setiap kategori.\n"
        "4. Pantau pengeluaran harian agar tidak melebihi batas.\n"
        "5. FinSmart dapat mengingatkan kamu saat mendekati batas anggaran."
    ),
    "kategori pengeluaran": (
        "Kategori Pengeluaran di FinSmart:\n"
        "FinSmart mengklasifikasikan transaksi ke 10 kategori:\n"
        "1. Food | 2. Grocery | 3. Transport | 4. Entertainment | 5. Healthcare\n"
        "6. Electronics | 7. Online Shopping | 8. Travel | 9. Clothing | 10. Bills\n\n"
        "Kategori ditentukan otomatis oleh AI saat kamu input transaksi."
    ),
    "investasi untuk pemula": (
        "Panduan Investasi untuk Pemula:\n"
        "Sebelum investasi, pastikan sudah:\n"
        "- Punya dana darurat (3-6x pengeluaran bulanan)\n"
        "- Tidak punya hutang konsumtif\n"
        "- Tabungan rutin sudah berjalan\n\n"
        "Pilihan investasi untuk pemula:\n"
        "- Reksa Dana Pasar Uang — risiko rendah, cocok untuk pemula\n"
        "- Deposito — aman, bunga pasti\n"
        "- Obligasi Negara (ORI/SBR) — dijamin pemerintah\n\n"
        "Cek fitur Kesiapan Investasi di FinSmart untuk analisis profil keuanganmu."
    ),
    "cara mengurangi pengeluaran": (
        "Tips Mengurangi Pengeluaran:\n"
        "1. Identifikasi kategori pengeluaran terbesar dari dashboard FinSmart.\n"
        "2. Bedakan kebutuhan (harus) vs keinginan (bisa ditunda).\n"
        "3. Kurangi frekuensi makan di luar — masak sendiri bisa hemat 60%.\n"
        "4. Manfaatkan promo dan cashback saat belanja online.\n"
        "5. Batalkan langganan yang jarang dipakai."
    ),
    "target tabungan": (
        "Cara Menetapkan Target Tabungan:\n"
        "1. Tentukan tujuan spesifik: liburan, DP rumah, dana darurat, dll.\n"
        "2. Hitung nominal yang dibutuhkan dan tenggat waktunya.\n"
        "3. Bagi dengan jumlah bulan untuk mendapatkan target bulanan.\n"
        "4. Set target di fitur Sistem Anggaran FinSmart.\n"
        "5. FinSmart akan memberi notifikasi jika kamu on-track."
    ),
    "rekomendasi keuangan": (
        "Rekomendasi Keuangan Personal:\n"
        "Berdasarkan data keuanganmu, FinSmart memberikan rekomendasi:\n"
        "- Analisis pola pengeluaran bulanan\n"
        "- Kategori mana yang paling banyak menguras anggaran\n"
        "- Rekomendasi batas anggaran per kategori\n\n"
        "Masuk ke menu Analisis dan Rekomendasi AI untuk melihat insight personalmu."
    ),
    "pengertian literasi keuangan": (
        "Apa itu Literasi Keuangan?\n"
        "Literasi keuangan adalah kemampuan memahami dan menggunakan\n"
        "berbagai konsep keuangan secara efektif, meliputi:\n"
        "- Mengelola pendapatan dan pengeluaran\n"
        "- Merencanakan tabungan dan investasi\n"
        "- Memahami produk keuangan (kredit, asuransi, investasi)\n"
        "- Membuat keputusan keuangan yang bijak\n\n"
        "Data OJK 2024: literasi keuangan generasi muda Indonesia masih di bawah 50%."
    ),
}
RESPONSE_DEFAULT = (
    "Maaf, saya belum bisa menjawab pertanyaan tersebut.\n"
    "Kamu bisa tanya tentang: tips menabung, mengatur anggaran, "
    "kategori pengeluaran, investasi pemula, atau target tabungan."
)

# ──────────────────────────────────────────────
# INSTRUMEN INVESTASI
# ──────────────────────────────────────────────
INSTRUMEN = {
    "konservatif": ["Tabungan berjangka / Deposito", "Obligasi Negara (ORI/SBR)", "Reksa Dana Pasar Uang"],
    "moderat":     ["Reksa Dana Pendapatan Tetap", "ORI/SBR", "P2P Lending OJK"],
    "agresif":     ["Reksa Dana Saham", "Saham blue-chip (LQ45)", "ETF / Index Fund"],
}

# ──────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ──────────────────────────────────────────────

class TransaksiInput(BaseModel):
    """
    Input endpoint /classify — sesuai dataset baru DS.
    Terima nilai string asli, API yang akan encode.
    """
    Amount: float          # nominal transaksi (Rp, raw — akan di-scale otomatis)
    Payment_Method: str    # "Credit Card" | "Debit Card" | "Net Banking" | "UPI"
    Week_Day: str          # "Monday" | "Tuesday" | ... | "Sunday"
    Month: str             # "January" | "February" | ... | "December"
    Time_Of_Day: str       # "Morning" | "Afternoon" | "Evening" | "Night"
    MerchantName: str      # nama merchant (lihat /valid-values)
    Day: int               # tanggal 1–31

class BehaviorInput(BaseModel):
    """
    Input endpoint /behavior — sesuai dataset DS (financial_type_classifier.pkl).
    Semua nilai dalam Rupiah.
    """
    Income: float           # total pemasukan bulanan
    Needs: float            # total kebutuhan primer (bahan pokok, tempat tinggal, dll)
    Wants: float            # total kebutuhan sekunder (hiburan, lifestyle, dll)
    Savings: float          # total tabungan bulan ini
    Total_Spending: float   # total pengeluaran (Needs + Wants)
    Financial_Balance: float  # sisa income setelah pengeluaran (Income - Total_Spending)

class PrediksiSpendingInput(BaseModel):
    """
    Input endpoint /predict-spending.
    histori: list of dict, minimal 3 bulan terakhir.
    Setiap dict berisi: income, total_spending, savings,
    dan pengeluaran per kategori (Food & Dining, Transportation, dll)
    """
    histori: list

    class Config:
        json_schema_extra = {
            "example": {
                "histori": [
                    {
                        "income": 5000000, "total_spending": 3200000, "savings": 1800000,
                        "Food & Dining": 800000, "Transportation": 400000, "Shopping": 500000,
                        "Groceries": 300000, "Bills & Utilities": 400000, "Entertainment": 200000,
                        "Health": 100000, "Education": 200000, "Others": 300000
                    },
                    {
                        "income": 5000000, "total_spending": 3500000, "savings": 1500000,
                        "Food & Dining": 900000, "Transportation": 450000, "Shopping": 600000,
                        "Groceries": 350000, "Bills & Utilities": 400000, "Entertainment": 250000,
                        "Health": 150000, "Education": 200000, "Others": 200000
                    },
                    {
                        "income": 5000000, "total_spending": 3800000, "savings": 1200000,
                        "Food & Dining": 1000000, "Transportation": 500000, "Shopping": 700000,
                        "Groceries": 400000, "Bills & Utilities": 400000, "Entertainment": 300000,
                        "Health": 100000, "Education": 200000, "Others": 200000
                    }
                ]
            }
        }

class ChatInput(BaseModel):
    pertanyaan: str

class RekomendasiInput(BaseModel):
    """
    Input endpoint /rekomendasi — BLR + SR sesuai rumus resmi DS.
    """
    tabungan_total: float    # total tabungan yang dimiliki saat ini
    total_pengeluaran_bulanan: float  # rata-rata pengeluaran per bulan
    tabungan_bulanan: float  # jumlah yang berhasil ditabung per bulan
    income_bulanan: float    # total pemasukan per bulan


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def preprocess_transaksi(data: TransaksiInput) -> np.ndarray:
    """
    Preprocessing input transaksi:
    - Encode fitur kategorik sesuai mapping DS
    - Scale Amount dengan scaler yang sudah di-fit
    """
    # Validasi & encode Payment_Method
    if data.Payment_Method not in PAYMENT_METHOD_MAP:
        raise HTTPException(422, f"Payment_Method tidak valid: '{data.Payment_Method}'. "
                            f"Pilihan: {list(PAYMENT_METHOD_MAP.keys())}")

    # Validasi & encode Week_Day
    if data.Week_Day not in WEEK_DAY_MAP:
        raise HTTPException(422, f"Week_Day tidak valid: '{data.Week_Day}'. "
                            f"Pilihan: {list(WEEK_DAY_MAP.keys())}")

    # Validasi & encode Month
    if data.Month not in MONTH_MAP:
        raise HTTPException(422, f"Month tidak valid: '{data.Month}'. "
                            f"Pilihan: {list(MONTH_MAP.keys())}")

    # Validasi & encode Time_Of_Day
    if data.Time_Of_Day not in TIME_OF_DAY_MAP:
        raise HTTPException(422, f"Time_Of_Day tidak valid: '{data.Time_Of_Day}'. "
                            f"Pilihan: {list(TIME_OF_DAY_MAP.keys())}")

    # Validasi & encode MerchantName
    if data.MerchantName not in MERCHANT_MAP:
        raise HTTPException(422, f"MerchantName tidak valid: '{data.MerchantName}'. "
                            f"Gunakan GET /valid-values untuk melihat daftar merchant.")

    # Validasi Day
    if not (1 <= data.Day <= 31):
        raise HTTPException(422, f"Day harus antara 1–31, diterima: {data.Day}")

    # Scale Amount
    amount_scaled = scaler.transform([[data.Amount]])[0][0]

    # Susun fitur sesuai urutan training
    X = np.array([[
        amount_scaled,
        PAYMENT_METHOD_MAP[data.Payment_Method],
        WEEK_DAY_MAP[data.Week_Day],
        MONTH_MAP[data.Month],
        TIME_OF_DAY_MAP[data.Time_Of_Day],
        MERCHANT_MAP[data.MerchantName],
        data.Day
    ]])

    return X


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "FinSmart AI Service",
        "version": "3.0.0",
        "endpoints": [
            "POST /classify    — klasifikasi kategori transaksi",
            "POST /behavior    — klasifikasi perilaku keuangan (Hemat/Normal/Boros)",
            "POST /rekomendasi — kesiapan investasi (BLR + SR)",
            "POST /finbot/chat — chatbot keuangan",
            "GET  /model-info  — info model",
            "GET  /valid-values — nilai valid tiap fitur",
        ]
    }


@app.post("/classify", tags=["Klasifikasi"])
def classify(transaksi: TransaksiInput):
    """
    Klasifikasi kategori pengeluaran dari data transaksi.

    Input: data transaksi (Amount, Payment_Method, Week_Day, Month, Time_Of_Day, MerchantName, Day)
    Output: kategori prediksi + confidence + top-3 probabilitas
    """
    X     = preprocess_transaksi(transaksi)
    proba = xgb_model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))

    kategori = CATEGORY_CLASSES[idx]
    top3 = dict(sorted(
        {k: round(float(p), 4) for k, p in zip(CATEGORY_CLASSES, proba)}.items(),
        key=lambda x: x[1], reverse=True
    )[:3])

    return {
        "kategori"  : kategori,
        "confidence": round(float(proba[idx]) * 100, 2),
        "top3_prob" : top3
    }


@app.post("/behavior", tags=["Behavior"])
def classify_behavior(data: BehaviorInput):
    """
    Klasifikasi perilaku keuangan pengguna (Hemat / Normal / Boros).

    Menggunakan model Decision Tree dari tim Data Scientist (financial_type_classifier.pkl).
    Berdasarkan prinsip 50/30/20.

    Input: Income, Needs, Wants, Savings, Total_Spending, Financial_Balance
    Output: tipe perilaku + rasio keuangan
    """
    # Hitung rasio untuk informasi tambahan
    savings_ratio = data.Savings / data.Income if data.Income > 0 else 0
    wants_ratio   = data.Wants   / data.Income if data.Income > 0 else 0
    needs_ratio   = data.Needs   / data.Income if data.Income > 0 else 0

    # Susun fitur sesuai urutan training model DS
    # ['Income', 'Needs', 'Wants', 'Savings', 'Total_Spending', 'Financial_Balance']
    if behavior_model is None:
        raise HTTPException(503, "Model behavior tidak tersedia saat ini.")

    X = np.array([[
        data.Income,
        data.Needs,
        data.Wants,
        data.Savings,
        data.Total_Spending,
        data.Financial_Balance
    ]])

    tipe = behavior_model.predict(X)[0]

    # Skor kepercayaan jika model support predict_proba
    confidence = None
    try:
        proba_behavior = behavior_model.predict_proba(X)[0]
        classes        = behavior_model.classes_
        proba_dict     = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba_behavior)}
        confidence     = proba_dict
    except Exception:
        pass

    return {
        "tipe_keuangan" : tipe,
        "confidence_pct": confidence,
        "rasio": {
            "savings_ratio_pct": round(savings_ratio * 100, 2),
            "wants_ratio_pct"  : round(wants_ratio * 100, 2),
            "needs_ratio_pct"  : round(needs_ratio * 100, 2),
        },
        "interpretasi": {
            "hemat" : "Savings > 20% DAN Wants < 30% — tabungan tinggi, lifestyle terkontrol",
            "normal": "Mendekati prinsip 50/30/20 — pengeluaran dan tabungan relatif seimbang",
            "boros" : "Savings < 20% DAN Wants > 30% — lifestyle tinggi, tabungan rendah"
        }.get(tipe, "-")
    }


@app.post("/rekomendasi", tags=["Rekomendasi"])
def rekomendasi_investasi(body: RekomendasiInput):
    """
    Analisis kesiapan investasi berdasarkan rumus resmi DS:
    - BLR (Basic Liquidity Ratio) = Tabungan Total / Total Pengeluaran Bulanan
    - SR  (Savings Ratio)         = Tabungan Bulanan / Income Bulanan x 100%

    Status:
    - Not Ready        : BLR < 1, atau BLR 1-3 & SR < 10%
    - Moderately Ready : BLR 1-3 & SR >= 10%, atau BLR > 3 & SR < 10%
    - Ready            : BLR > 3 & SR >= 10%
    """
    # Hitung BLR
    blr = (body.tabungan_total / body.total_pengeluaran_bulanan
           if body.total_pengeluaran_bulanan > 0 else 0)

    # Hitung SR
    sr = (body.tabungan_bulanan / body.income_bulanan * 100
          if body.income_bulanan > 0 else 0)

    # Tentukan kondisi BLR
    if blr < 1:
        blr_kondisi = "Sangat rentan"
    elif blr <= 3:
        blr_kondisi = "Kurang aman"
    elif blr <= 6:
        blr_kondisi = "Cukup sehat"
    else:
        blr_kondisi = "Sangat siap"

    # Tentukan kondisi SR
    if sr < 10:
        sr_kondisi = "Kurang sehat"
    elif sr <= 20:
        sr_kondisi = "Cukup"
    else:
        sr_kondisi = "Baik"

    # Tentukan status kesiapan investasi (tabel resmi DS)
    if blr < 1:
        status = "Not Ready"
    elif blr <= 3 and sr < 10:
        status = "Not Ready"
    elif blr <= 3 and sr >= 10:
        status = "Moderately Ready"
    elif blr > 3 and sr < 10:
        status = "Moderately Ready"
    else:  # blr > 3 and sr >= 10
        status = "Ready"

    # Profil risiko & instrumen berdasarkan status
    if status == "Ready":
        profil    = "agresif" if sr > 20 else "moderat"
    elif status == "Moderately Ready":
        profil    = "moderat"
    else:
        profil    = "konservatif"

    return {
        "blr": {
            "nilai"  : round(blr, 2),
            "kondisi": blr_kondisi,
            "rumus"  : "Tabungan Total / Total Pengeluaran Bulanan"
        },
        "sr": {
            "nilai_pct": round(sr, 2),
            "kondisi"  : sr_kondisi,
            "rumus"    : "Tabungan Bulanan / Income Bulanan x 100%"
        },
        "status_kesiapan"  : status,
        "profil_risiko"    : profil,
        "rekomendasi_instrumen": INSTRUMEN.get(profil, []),
        "pesan": {
            "Not Ready"       : "Fokus dulu membangun dana darurat dan meningkatkan tabungan sebelum investasi.",
            "Moderately Ready": "Kondisi keuangan mulai stabil. Bisa mulai investasi konservatif sambil perkuat tabungan.",
            "Ready"           : "Kondisi keuangan sehat! Siap untuk mulai berinvestasi sesuai profil risiko."
        }.get(status, "")
    }



@app.post("/predict-spending", tags=["Prediksi"])
def predict_spending(body: PrediksiSpendingInput):
    """
    Prediksi total pengeluaran pengguna bulan berikutnya menggunakan LSTM.

    Input : histori pengeluaran minimal 3 bulan terakhir
    Output: prediksi total pengeluaran bulan depan + insight
    """
    if not LSTM_AVAILABLE or lstm_model is None:
        raise HTTPException(503, "Model LSTM tidak tersedia saat ini.")

    histori = list(body.histori)
    if len(histori) < LSTM_SEQUENCE_LENGTH:
        raise HTTPException(
            422,
            f"Histori minimal {LSTM_SEQUENCE_LENGTH} bulan, diterima {len(histori)} bulan."
        )

    for h in histori:
        for feat in LSTM_FEATURES:
            if feat not in h:
                raise HTTPException(422, f"Field '{feat}' tidak ditemukan di histori.")

    histori = histori[-LSTM_SEQUENCE_LENGTH:]
    raw     = np.array([[h[f] for f in LSTM_FEATURES] for h in histori])
    scaled  = scaler_lstm.transform(raw)
    X       = scaled.reshape(1, LSTM_SEQUENCE_LENGTH, len(LSTM_FEATURES))

    y_scaled = lstm_model.predict(X, verbose=0)[0][0]

    dummy = np.zeros((1, len(LSTM_FEATURES)))
    target_idx = LSTM_FEATURES.index("total_spending")
    dummy[0, target_idx] = y_scaled
    y_actual = scaler_lstm.inverse_transform(dummy)[0, target_idx]

    last_spending = histori[-1]["total_spending"]
    selisih       = float(y_actual) - float(last_spending)
    pct_change    = (selisih / float(last_spending)) * 100 if last_spending > 0 else 0

    return {
        "prediksi_spending_bulan_depan": round(float(y_actual)),
        "spending_bulan_lalu"          : round(float(last_spending)),
        "selisih"                      : round(float(selisih)),
        "perubahan_pct"                : round(float(pct_change), 2),
        "trend"                        : "naik" if selisih > 0 else "turun",
        "insight": (
            f"Prediksi pengeluaran bulan depan Rp {y_actual:,.0f}. "
            f"{'Naik' if selisih > 0 else 'Turun'} {abs(pct_change):.1f}% dari bulan lalu."
        )
    }

@app.post("/finbot/chat", tags=["FinBot"])
def finbot_chat(body: ChatInput):
    """
    Chatbot keuangan FinBot berbasis Zero-Shot Classification (Hugging Face).

    Input : pertanyaan keuangan dalam bahasa Indonesia
    Output: intent yang terdeteksi + respons edukasi keuangan
    """
    if not body.pertanyaan.strip():
        raise HTTPException(422, "Pertanyaan tidak boleh kosong.")

    if finbot_clf is None:
        return {
            "intent"    : "tidak_tersedia",
            "confidence": 0,
            "respons"   : "FinBot sedang tidak tersedia di server ini. Silakan gunakan aplikasi FinSmart secara langsung."
        }
    result     = finbot_clf(
        body.pertanyaan,
        candidate_labels=INTENT_LABELS,
        hypothesis_template="Pertanyaan ini tentang {}."
    )
    intent     = result["labels"][0]
    confidence = result["scores"][0]
    respons    = RESPONSE_DB.get(intent, RESPONSE_DEFAULT) if confidence >= 0.30 else RESPONSE_DEFAULT

    return {
        "intent"    : intent,
        "confidence": round(confidence * 100, 2),
        "respons"   : respons
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    """Informasi model yang sedang digunakan."""
    return metadata


@app.get("/valid-values", tags=["Info"])
def valid_values():
    """Daftar nilai valid untuk setiap fitur input endpoint /classify."""
    return {
        "Payment_Method": list(PAYMENT_METHOD_MAP.keys()),
        "Week_Day"      : list(WEEK_DAY_MAP.keys()),
        "Month"         : list(MONTH_MAP.keys()),
        "Time_Of_Day"   : list(TIME_OF_DAY_MAP.keys()),
        "MerchantName"  : list(MERCHANT_MAP.keys()),
        "Day"           : "integer 1–31",
        "Amount"        : "float, nominal transaksi dalam Rupiah"
    }
