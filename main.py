"""
FinSmart AI Service — FastAPI Basic
Coding Camp 2026 | CC26-PSU407 | AI Engineer: Muhammad Syaiful

Jalankan: python -m uvicorn main:app --port 8000
Docs    : http://localhost:8000/docs
"""

import numpy as np
import pickle
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="FinSmart AI Service",
    description="API klasifikasi kategori pengeluaran berbasis AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load("xgb_model.pkl")

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_metadata.json") as f:
    metadata = json.load(f)

PAYMENT_METHOD_MAP = {"Credit Card": 0, "Debit Card": 1, "Net Banking": 2, "UPI": 3}
WEEK_DAY_MAP       = {"Friday": 0, "Monday": 1, "Saturday": 2, "Sunday": 3,
                      "Thursday": 4, "Tuesday": 5, "Wednesday": 6}
TIME_OF_DAY_MAP    = {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3}
MONTH_MAP          = {"April": 0, "August": 1, "December": 2, "February": 3,
                      "January": 4, "July": 5, "June": 6, "March": 7,
                      "May": 8, "November": 9, "October": 10, "September": 11}
MERCHANT_MAP       = {
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
CATEGORY_CLASSES = [
    "Bills", "Clothing", "Electronics", "Entertainment",
    "Food", "Grocery", "Healthcare", "Online Shopping", "Transport", "Travel"
]


class TransaksiInput(BaseModel):
    Amount: float
    Payment_Method: str
    Week_Day: str
    Month: str
    Time_Of_Day: str
    MerchantName: str
    Day: int

    class Config:
        json_schema_extra = {
            "example": {
                "Amount": 250000,
                "Payment_Method": "Credit Card",
                "Week_Day": "Saturday",
                "Month": "May",
                "Time_Of_Day": "Evening",
                "MerchantName": "Amazon",
                "Day": 10
            }
        }


def preprocess(data: TransaksiInput) -> np.ndarray:
    if data.Payment_Method not in PAYMENT_METHOD_MAP:
        raise HTTPException(422, f"Payment_Method tidak valid: '{data.Payment_Method}'. "
                            f"Pilihan: {list(PAYMENT_METHOD_MAP.keys())}")
    if data.Week_Day not in WEEK_DAY_MAP:
        raise HTTPException(422, f"Week_Day tidak valid: '{data.Week_Day}'. "
                            f"Pilihan: {list(WEEK_DAY_MAP.keys())}")
    if data.Month not in MONTH_MAP:
        raise HTTPException(422, f"Month tidak valid: '{data.Month}'. "
                            f"Pilihan: {list(MONTH_MAP.keys())}")
    if data.Time_Of_Day not in TIME_OF_DAY_MAP:
        raise HTTPException(422, f"Time_Of_Day tidak valid: '{data.Time_Of_Day}'. "
                            f"Pilihan: {list(TIME_OF_DAY_MAP.keys())}")
    if data.MerchantName not in MERCHANT_MAP:
        raise HTTPException(422, f"MerchantName tidak valid: '{data.MerchantName}'. "
                            f"Gunakan GET /valid-values untuk daftar merchant.")
    if not (1 <= data.Day <= 31):
        raise HTTPException(422, f"Day harus antara 1-31.")

    amount_scaled = scaler.transform([[data.Amount]])[0][0]

    return np.array([[
        amount_scaled,
        PAYMENT_METHOD_MAP[data.Payment_Method],
        WEEK_DAY_MAP[data.Week_Day],
        MONTH_MAP[data.Month],
        TIME_OF_DAY_MAP[data.Time_Of_Day],
        MERCHANT_MAP[data.MerchantName],
        data.Day
    ]])


@app.get("/", tags=["Health"])
def root():
    return {
        "status" : "ok",
        "service": "FinSmart AI Service",
        "version": "2.0.0",
        "docs"   : "/docs"
    }


@app.post("/classify", tags=["Klasifikasi"])
def classify(transaksi: TransaksiInput):
    X     = preprocess(transaksi)
    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))

    kategori = CATEGORY_CLASSES[idx]
    top3     = dict(sorted(
        {k: round(float(p), 4) for k, p in zip(CATEGORY_CLASSES, proba)}.items(),
        key=lambda x: x[1], reverse=True
    )[:3])

    return {
        "kategori"  : kategori,
        "confidence": round(float(proba[idx]) * 100, 2),
        "top3_prob" : top3
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    return metadata


@app.get("/valid-values", tags=["Info"])
def valid_values():
    return {
        "Payment_Method": list(PAYMENT_METHOD_MAP.keys()),
        "Week_Day"      : list(WEEK_DAY_MAP.keys()),
        "Month"         : list(MONTH_MAP.keys()),
        "Time_Of_Day"   : list(TIME_OF_DAY_MAP.keys()),
        "MerchantName"  : list(MERCHANT_MAP.keys()),
        "Day"           : "integer 1-31",
        "Amount"        : "float, nominal transaksi"
    }


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
