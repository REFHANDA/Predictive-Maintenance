"""
utils/helper.py
Kumpulan fungsi bantu umum:
- Validasi input (Robustness Check, lihat Soal Umpan Balik no. 5)
- Membangun matriks keputusan SPK dari hasil prediksi ML + data statis mesin (Langkah 1 integrasi)
- Anonimisasi data sensitif sebelum ditampilkan di dashboard publik (Soal Umpan Balik no. 7)
- Fungsi format angka & konversi untuk download CSV
"""

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Suhu", "Vibrasi", "Tekanan", "Usia_Mesin", "Jam_Operasi", "Beban"]


def validate_input(values: Dict[str, float], valid_ranges: Dict[str, list]) -> List[str]:
    """
    Memvalidasi nilai input pengguna terhadap rentang wajar (valid_ranges, dari model_metadata.json).

    Mengembalikan list peringatan (string). List kosong = semua input dalam rentang aman.
    Tidak menghentikan eksekusi (sistem tetap menampilkan prediksi), tapi memberi peringatan
    eksplisit ke pengguna bahwa hasil berada di luar zona training -> potensi model drift.
    """
    warnings = []
    for feature, value in values.items():
        if feature not in valid_ranges:
            continue
        low, high = valid_ranges[feature]
        if value < low or value > high:
            warnings.append(
                f"⚠️ Nilai {feature} = {value} berada di luar rentang data training "
                f"({low}–{high}). Prediksi pada area ini berisiko tidak akurat (ekstrapolasi/drift)."
            )
    return warnings


def build_decision_matrix(
    machines: List[dict],
    risk_predictions: Dict[str, float],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Membangun matriks keputusan SPK dengan menggabungkan:
    - Kolom Risiko -> hasil prediksi ML (dinamis, berubah sesuai slider What-If)
    - Kolom Biaya_Perawatan dan Efisiensi -> data statis tiap mesin

    Parameters
    ----------
    machines : list of dict, contoh:
        [{"nama": "Mesin A", "biaya_perawatan": 12, "efisiensi": 85}, ...]
    risk_predictions : dict {nama_mesin: skor_risiko_dari_model_ml}

    Returns
    -------
    matrix (np.ndarray), alternative_names (list), criteria_names (list)
    """
    criteria_names = ["Risiko_ML", "Biaya_Perawatan", "Efisiensi"]
    alternative_names = [m["nama"] for m in machines]

    rows = []
    for m in machines:
        risiko = risk_predictions.get(m["nama"], 0.0)
        rows.append([risiko, m["biaya_perawatan"], m["efisiensi"]])

    matrix = np.array(rows, dtype=float)
    return matrix, alternative_names, criteria_names


def anonymize_dataframe(df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
    """
    Menghapus/mengganti kolom yang berpotensi melanggar privasi (misal: ID Operator,
    Nama Teknisi, Lokasi GPS presisi) sebelum data ditampilkan pada demo publik.

    Strategi: drop kolom sensitif jika ada di DataFrame, daripada menampilkan nilai asli.
    """
    cols_to_drop = [c for c in sensitive_columns if c in df.columns]
    return df.drop(columns=cols_to_drop)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Mengonversi DataFrame menjadi bytes CSV untuk tombol download Streamlit."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def compute_delta(baseline_value: float, current_value: float) -> Tuple[float, float]:
    """
    Menghitung delta absolut dan delta persentase antara skenario baseline dan skenario
    What-If saat ini. Dipakai pada fitur "Baseline vs Current".
    """
    delta_abs = current_value - baseline_value
    delta_pct = (delta_abs / baseline_value * 100) if baseline_value != 0 else 0.0
    return round(delta_abs, 2), round(delta_pct, 2)


def confidence_message(rmse: float, prediction: float) -> str:
    """
    Menghasilkan pesan ketidakpastian model berbasis RMSE training, untuk ditampilkan
    di UI sebagai interval kepercayaan kasar (Soal Umpan Balik no. 8).
    """
    lower = max(0, round(prediction - rmse, 2))
    upper = min(100, round(prediction + rmse, 2))
    return (
        f"Estimasi skor risiko: **{prediction:.2f}** "
        f"(rentang ketidakpastian kira-kira {lower}–{upper}, berdasarkan RMSE model = {rmse:.2f})."
    )
