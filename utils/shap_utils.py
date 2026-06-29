"""
utils/shap_utils.py
Modul bantu untuk Transparansi/Explainability (XAI) menggunakan SHAP.

Berisi fungsi-fungsi untuk:
- Membangun SHAP explainer dari model yang sudah dilatih
- Menghitung SHAP values untuk satu baris input (real-time, dari slider What-If)
- Menyiapkan figure waterfall plot (kontribusi fitur untuk 1 prediksi)
- Menyiapkan figure summary plot (kepentingan fitur secara global)
- Mengambil kontribusi fitur dalam bentuk tabel (untuk narasi/justifikasi)
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def build_explainer(model) -> shap.TreeExplainer:
    """Membangun SHAP TreeExplainer untuk model berbasis pohon (RandomForest)."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X_scaled: np.ndarray):
    """Menghitung SHAP values untuk input yang sudah di-scale."""
    return explainer(X_scaled)


def get_waterfall_figure(shap_values, index: int = 0, max_display: int = 10):
    """
    Mengembalikan matplotlib Figure berisi waterfall plot untuk satu baris prediksi.
    Dipakai untuk menjawab pertanyaan "Mengapa hasilnya demikian?" pada 1 input spesifik.
    """
    fig = plt.figure()
    shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def get_summary_figure(shap_values, feature_names: List[str], max_display: int = 10):
    """
    Mengembalikan matplotlib Figure berisi summary plot (bar) untuk kepentingan
    fitur secara global/rata-rata, biasanya dihitung dari sampel data test.
    """
    fig = plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def get_contribution_table(shap_values, feature_names: List[str], index: int = 0) -> pd.DataFrame:
    """
    Mengembalikan tabel kontribusi fitur (nilai SHAP) untuk satu prediksi,
    diurutkan dari kontribusi absolut terbesar. Cocok untuk narasi tekstual:
    "karena fitur X memberikan kontribusi positif/negatif sebesar Y".
    """
    values = shap_values[index].values
    base_value = shap_values[index].base_values

    df = pd.DataFrame({
        "Fitur": feature_names,
        "Nilai_SHAP": values,
    })
    df["Arah"] = np.where(df["Nilai_SHAP"] >= 0, "Menaikkan Risiko", "Menurunkan Risiko")
    df["Kontribusi_Absolut"] = df["Nilai_SHAP"].abs()
    df = df.sort_values("Kontribusi_Absolut", ascending=False).reset_index(drop=True)
    df.attrs["base_value"] = float(base_value)
    return df


def explain_top_feature(shap_values, feature_names: List[str], index: int = 0) -> str:
    """
    Menghasilkan kalimat narasi otomatis untuk fitur dengan kontribusi terbesar.
    Membantu menjawab pertanyaan black-box seperti pada Soal Umpan Balik no. 4.
    """
    table = get_contribution_table(shap_values, feature_names, index)
    top = table.iloc[0]
    arah = "menaikkan" if top["Nilai_SHAP"] >= 0 else "menurunkan"
    return (
        f"Fitur '{top['Fitur']}' memberikan kontribusi paling besar, "
        f"{arah} skor risiko sebesar {abs(top['Nilai_SHAP']):.2f} poin."
    )
