"""
generate_data.py
Membuat dataset sintetis machine_data.csv untuk Project Smart Predictive Maintenance.

Fitur (kriteria input):
- Suhu (Temperature, Celsius)
- Vibrasi (Vibration, mm/s)
- Tekanan (Pressure, bar)
- Usia_Mesin (Machine Age, tahun)
- Jam_Operasi (Operating Hours per minggu)
- Beban (Load, %)

Target:
- Risk_Score (0-100) -> semakin tinggi semakin berisiko gagal/rusak
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

suhu = np.random.normal(65, 15, N).clip(20, 120)            # derajat C
vibrasi = np.random.normal(3.5, 1.5, N).clip(0.1, 12)        # mm/s
tekanan = np.random.normal(5.5, 1.8, N).clip(0.5, 15)        # bar
usia_mesin = np.random.uniform(0, 20, N)                     # tahun
jam_operasi = np.random.normal(60, 20, N).clip(5, 168)        # jam/minggu
beban = np.random.normal(70, 20, N).clip(5, 100)              # %

noise = np.random.normal(0, 5, N)

# Formula sintetis: kombinasi non-linear supaya RandomForest punya pola menarik
risk_score = (
    0.20 * (suhu - 20) +
    3.0 * vibrasi +
    1.2 * (tekanan - 0.5) +
    0.9 * usia_mesin +
    0.10 * jam_operasi +
    0.15 * beban +
    0.001 * (suhu - 60) ** 2 +
    noise
) - 5

risk_score = np.clip(risk_score, 0, 100)

df = pd.DataFrame({
    "Suhu": suhu.round(2),
    "Vibrasi": vibrasi.round(2),
    "Tekanan": tekanan.round(2),
    "Usia_Mesin": usia_mesin.round(2),
    "Jam_Operasi": jam_operasi.round(1),
    "Beban": beban.round(1),
    "Risk_Score": risk_score.round(2),
})

df.to_csv("/home/claude/Predictive_Maintenance/data/machine_data.csv", index=False)
print("Dataset tersimpan:", df.shape)
print(df.describe())
