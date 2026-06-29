"""
app.py
Smart Predictive Maintenance - Dashboard Streamlit (Project UAS Minggu 16)

Alur integrasi (sesuai lecture note Minggu 16):
Slider (UI) -> Scaler -> Model ML (RandomForest) -> Risiko Mesin
            -> Matriks Keputusan SPK -> SAW (dengan bobot AHP) -> Ranking Prioritas Perawatan
            -> SHAP -> Penjelasan kontribusi fitur (XAI)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.helper import (
    build_decision_matrix,
    compute_delta,
    confidence_message,
    dataframe_to_csv_bytes,
    validate_input,
)
from utils.saw import run_saw
from utils.shap_utils import (
    build_explainer,
    compute_shap_values,
    explain_top_feature,
    get_contribution_table,
    get_waterfall_figure,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLUMNS = ["Suhu", "Vibrasi", "Tekanan", "Usia_Mesin", "Jam_Operasi", "Beban"]

st.set_page_config(
    page_title="Smart Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Load model, scaler, metadata (cache supaya tidak reload setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_resource
def get_explainer(_model):
    return build_explainer(_model)


model, scaler, metadata = load_artifacts()
explainer = get_explainer(model)
VALID_RANGES = metadata["valid_ranges"]
RMSE = metadata["rmse"]

# Data statis dua mesin pembanding (selain mesin yang sedang diuji via slider).
# Pada implementasi nyata, ini bisa diambil dari database aset/maintenance log.
REFERENCE_MACHINES = {
    "Mesin B": {
        "features": {"Suhu": 58, "Vibrasi": 2.8, "Tekanan": 4.5,
                     "Usia_Mesin": 4, "Jam_Operasi": 50, "Beban": 60},
        "biaya_perawatan": 8,   # skala 1-20, makin kecil makin murah (cost)
        "efisiensi": 88,        # skala 0-100, makin besar makin baik (benefit)
    },
    "Mesin C": {
        "features": {"Suhu": 82, "Vibrasi": 5.4, "Tekanan": 6.8,
                     "Usia_Mesin": 12, "Jam_Operasi": 95, "Beban": 85},
        "biaya_perawatan": 15,
        "efisiensi": 70,
    },
}

if "baseline" not in st.session_state:
    st.session_state.baseline = None


# ----------------------------------------------------------------------------
# SIDEBAR — Input Slider (Live Intervention / What-If)
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Input Skenario — Mesin A (Slider What-If)")
st.sidebar.caption("Geser slider untuk mensimulasikan kondisi mesin secara real-time.")

suhu = st.sidebar.slider("Suhu (°C)", 0, 200, 65)
vibrasi = st.sidebar.slider("Vibrasi (mm/s)", 0.0, 15.0, 3.5, step=0.1)
tekanan = st.sidebar.slider("Tekanan (bar)", 0.0, 20.0, 5.5, step=0.1)
usia_mesin = st.sidebar.slider("Usia Mesin (tahun)", 0, 30, 5)
jam_operasi = st.sidebar.slider("Jam Operasi (jam/minggu)", 0, 200, 60)
beban = st.sidebar.slider("Beban (%)", 0, 150, 70)

st.sidebar.divider()
biaya_perawatan_a = st.sidebar.number_input(
    "Biaya Perawatan Mesin A (skala 1-20, kecil=murah)", 1, 20, 10
)
efisiensi_a = st.sidebar.number_input(
    "Efisiensi Mesin A (skala 0-100)", 0, 100, 80
)

st.sidebar.divider()
if st.sidebar.button("📌 Simpan skenario ini sebagai Baseline"):
    st.session_state.baseline = {
        "Suhu": suhu, "Vibrasi": vibrasi, "Tekanan": tekanan,
        "Usia_Mesin": usia_mesin, "Jam_Operasi": jam_operasi, "Beban": beban,
    }
    st.sidebar.success("Baseline tersimpan!")

st.sidebar.divider()
st.sidebar.subheader("Bobot Kriteria SPK (hasil AHP)")
w_risk = st.sidebar.slider("Bobot Risiko (ML)", 0.0, 1.0, 0.5, step=0.05)
w_biaya = st.sidebar.slider("Bobot Biaya Perawatan", 0.0, 1.0, 0.2, step=0.05)
w_efisiensi = st.sidebar.slider("Bobot Efisiensi", 0.0, 1.0, 0.3, step=0.05)
total_w = w_risk + w_biaya + w_efisiensi


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🛠️ Smart Predictive Maintenance Simulator")
st.caption(
    "Studi Kasus: Bagaimana mengurangi kerugian gudang/produksi akibat kegagalan mesin "
    "yang tidak terdeteksi sejak dini?"
)

current_input = {
    "Suhu": suhu, "Vibrasi": vibrasi, "Tekanan": tekanan,
    "Usia_Mesin": usia_mesin, "Jam_Operasi": jam_operasi, "Beban": beban,
}

# ----------------------------------------------------------------------------
# VALIDASI INPUT (Robustness Check)
# ----------------------------------------------------------------------------
warnings = validate_input(current_input, VALID_RANGES)
if warnings:
    for w in warnings:
        st.warning(w)


# ----------------------------------------------------------------------------
# PREDIKSI ML
# ----------------------------------------------------------------------------
def predict_risk(features: dict) -> float:
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0]), X_scaled


risk_a, X_a_scaled = predict_risk(current_input)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Prediksi Risiko Mesin A", f"{risk_a:.2f} / 100")
with col2:
    if st.session_state.baseline:
        baseline_risk, _ = predict_risk(st.session_state.baseline)
        delta_abs, delta_pct = compute_delta(baseline_risk, risk_a)
        st.metric("Baseline vs Current", f"{baseline_risk:.2f} → {risk_a:.2f}",
                  delta=f"{delta_abs:+.2f} ({delta_pct:+.1f}%)")
    else:
        st.info("Belum ada baseline. Klik 'Simpan sebagai Baseline' di sidebar.")
with col3:
    st.caption(confidence_message(RMSE, risk_a))

st.divider()

# ----------------------------------------------------------------------------
# XAI — SHAP Explainability
# ----------------------------------------------------------------------------
st.subheader("🔍 Mengapa Hasilnya Demikian? (SHAP Explainability)")

shap_values_a = compute_shap_values(explainer, X_a_scaled)

col_shap1, col_shap2 = st.columns([1, 1])
with col_shap1:
    fig = get_waterfall_figure(shap_values_a, index=0)
    st.pyplot(fig, use_container_width=True)
with col_shap2:
    contrib_table = get_contribution_table(shap_values_a, FEATURE_COLUMNS, index=0)
    st.dataframe(contrib_table, use_container_width=True, hide_index=True)
    st.success(explain_top_feature(shap_values_a, FEATURE_COLUMNS, index=0))

st.divider()

# ----------------------------------------------------------------------------
# SPK — Integrasi SAW (Sintesis Akhir ML -> SPK)
# ----------------------------------------------------------------------------
st.subheader("📊 Rekomendasi Prioritas Perawatan (SAW)")

if abs(total_w - 1.0) > 1e-2:
    st.error(
        f"Total bobot kriteria saat ini = {total_w:.2f}, harus berjumlah 1.0. "
        "Sesuaikan slider bobot di sidebar."
    )
else:
    # Prediksi risiko untuk mesin pembanding (data statis -> tetap lewat model ML yang sama)
    risk_predictions = {"Mesin A": risk_a}
    for nama, data in REFERENCE_MACHINES.items():
        r, _ = predict_risk(data["features"])
        risk_predictions[nama] = r

    machines = [{"nama": "Mesin A", "biaya_perawatan": biaya_perawatan_a, "efisiensi": efisiensi_a}]
    for nama, data in REFERENCE_MACHINES.items():
        machines.append({
            "nama": nama,
            "biaya_perawatan": data["biaya_perawatan"],
            "efisiensi": data["efisiensi"],
        })

    matrix, alt_names, crit_names = build_decision_matrix(machines, risk_predictions)

    saw_result = run_saw(
        matrix=matrix,
        weights=[w_risk, w_biaya, w_efisiensi],
        criteria_types=["cost", "cost", "benefit"],
        alternative_names=alt_names,
        criteria_names=crit_names,
    )

    st.dataframe(saw_result, use_container_width=True, hide_index=True)

    top_alt = saw_result.iloc[0]["Alternatif"]
    st.success(f"🏆 Prioritas perawatan tertinggi: **{top_alt}** (Skor SAW tertinggi)")

    csv_bytes = dataframe_to_csv_bytes(saw_result)
    st.download_button(
        "⬇️ Download Hasil Simulasi (CSV)",
        data=csv_bytes,
        file_name="hasil_simulasi_saw.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    f"Model: RandomForestRegressor | RMSE training = {RMSE} | "
    "Pastikan data sensitif (ID operator, lokasi presisi) tidak ditampilkan saat demo publik."
)
