"""
utils/saw.py
Implementasi metode SAW (Simple Additive Weighting) untuk Sistem Pendukung Keputusan (SPK).

SAW digunakan untuk meranking alternatif (misal: Mesin A, Mesin B, Mesin C) berdasarkan
beberapa kriteria. Salah satu kriteria (Risiko) berasal dari hasil prediksi Machine Learning,
sedangkan kriteria lain (Biaya, Efisiensi) bersifat statis/diinput manual.

Jenis kriteria:
- "cost"    -> semakin kecil nilainya semakin baik (contoh: Risiko, Biaya Perawatan)
- "benefit" -> semakin besar nilainya semakin baik (contoh: Efisiensi)
"""

from typing import List, Literal

import numpy as np
import pandas as pd

CriteriaType = Literal["benefit", "cost"]


def normalize_matrix(matrix: np.ndarray, criteria_types: List[CriteriaType]) -> np.ndarray:
    """
    Normalisasi matriks keputusan sesuai aturan SAW.

    - benefit: r_ij = x_ij / max(x_j)
    - cost   : r_ij = min(x_j) / x_ij
    """
    matrix = matrix.astype(float)
    norm = np.zeros_like(matrix)

    for j, c_type in enumerate(criteria_types):
        col = matrix[:, j]

        # Hindari pembagian dengan nol
        col_safe = np.where(col == 0, 1e-9, col)

        if c_type == "benefit":
            max_val = col.max() if col.max() != 0 else 1e-9
            norm[:, j] = col / max_val
        elif c_type == "cost":
            min_val = col.min() if col.min() != 0 else 1e-9
            norm[:, j] = min_val / col_safe
        else:
            raise ValueError(f"Tipe kriteria tidak dikenal: {c_type}")

    return norm


def run_saw(
    matrix: np.ndarray,
    weights: List[float],
    criteria_types: List[CriteriaType],
    alternative_names: List[str],
    criteria_names: List[str],
) -> pd.DataFrame:
    """
    Menjalankan SAW lengkap: normalisasi -> pembobotan -> perankingan.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_alternatif, n_kriteria)
    weights : list bobot kriteria (harus berjumlah 1.0, biasanya hasil AHP)
    criteria_types : list "benefit"/"cost" untuk setiap kolom
    alternative_names : nama setiap alternatif/baris (misal ["Mesin A", "Mesin B"])
    criteria_names : nama setiap kriteria/kolom

    Returns
    -------
    pd.DataFrame berisi skor akhir dan ranking, urut dari skor tertinggi (terbaik).
    """
    weights = np.array(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0, atol=1e-2):
        raise ValueError(
            f"Total bobot harus mendekati 1.0 (hasil AHP). Total saat ini: {weights.sum():.3f}"
        )

    if matrix.shape[1] != len(weights) or matrix.shape[1] != len(criteria_types):
        raise ValueError("Jumlah kolom matriks, bobot, dan tipe kriteria harus sama.")

    norm_matrix = normalize_matrix(matrix, criteria_types)
    weighted_matrix = norm_matrix * weights
    scores = weighted_matrix.sum(axis=1)

    result = pd.DataFrame(matrix, columns=criteria_names)
    result.insert(0, "Alternatif", alternative_names)

    for j, name in enumerate(criteria_names):
        result[f"Normalisasi_{name}"] = norm_matrix[:, j].round(4)

    result["Skor_SAW"] = scores.round(4)
    result["Ranking"] = result["Skor_SAW"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("Ranking").reset_index(drop=True)

    return result


def ahp_weights_from_pairwise(pairwise_matrix: np.ndarray) -> np.ndarray:
    """
    Menghitung bobot kriteria menggunakan metode AHP (Analytic Hierarchy Process)
    dari matriks perbandingan berpasangan (pairwise comparison matrix).

    Menggunakan metode rata-rata kolom ternormalisasi (pendekatan sederhana,
    cukup akurat untuk matriks kecil dan konsisten dengan materi praktikum).
    """
    pairwise_matrix = pairwise_matrix.astype(float)
    col_sums = pairwise_matrix.sum(axis=0)
    norm_matrix = pairwise_matrix / col_sums
    weights = norm_matrix.mean(axis=1)
    return weights / weights.sum()
