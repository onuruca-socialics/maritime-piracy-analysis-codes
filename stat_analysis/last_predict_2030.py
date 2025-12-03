# -*- coding: utf-8 -*-
# ============================================================
#  KORSAN SALDIRILARI — ARIMA(0,2,3) ile 2030'a kadar tahmin
# ============================================================

# -------------------------------
# 0) KÜTÜPHANELER VE AYARLAR
# -------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Çıktı klasörü
OUT_DIR = Path("/Users/onuruca/Desktop/On going/pirates_data/outputs_arima_2030")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1) VERİYİ OKU (1993–2020) ve YILLIK TOPLAMI ÇIKAR
# -------------------------------
candidates = [
    "/Users/onuruca/Downloads/datelonglatwihtdate.csv"
]
csv_path = None
for p in candidates:
    if Path(p).exists():
        csv_path = Path(p)
        break

if csv_path is None:
    raise FileNotFoundError("Veri dosyası bulunamadı.")

# Dosyayı oku ve sütun adlarını sadeleştir
df_raw = pd.read_csv(csv_path)
df_raw.columns = [c.strip().lower() for c in df_raw.columns]

# -------------------------------
# 📌 DATE → DATETIME → YEAR DÖNÜŞÜMÜ
# -------------------------------
df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
df_raw["year"] = df_raw["date"].dt.year

# -------------------------------
# 1993–2020 arasını filtrele ve yıllık saldırı sayısını hesapla
# -------------------------------
yearly_counts = (
    df_raw[(df_raw["year"] >= 1993) & (df_raw["year"] <= 2020)]
      .groupby("year")
      .size()
      .reset_index(name="attack_count")
      .sort_values("year")
      .reset_index(drop=True)
)

# -------------------------------
# 2) 2021–2024 GERÇEK DEĞERLERİ EKLE
# -------------------------------
truth_2021_2024 = pd.DataFrame({
    "year": [2021, 2022, 2023, 2024],
    "attack_count": [135, 123, 127, 122]
})

ts_all = pd.concat([yearly_counts, truth_2021_2024], ignore_index=True).sort_values("year")
ts_all = ts_all.reset_index(drop=True)

# -------------------------------
# 3) TIME INDEX & FREQUENCY
# -------------------------------
ts_index = pd.to_datetime(ts_all["year"].astype(str) + "-12-31")
ts_series = pd.Series(ts_all["attack_count"].values, index=ts_index).asfreq("A-DEC")

# -------------------------------
# 4) TRAIN ARIMA(0,2,3)
# -------------------------------
order = (0, 2, 3)
model = ARIMA(ts_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
res = model.fit()

print(res.summary())



# -------------------------------
# 5) FORECAST 2025–2030
# -------------------------------
steps = 6
fc = res.get_forecast(steps=steps)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)

start_year = int(ts_all["year"].max()) + 1
years_forecast = list(range(start_year, start_year + steps))

forecast_df = pd.DataFrame({
    "year": years_forecast,
    "forecast": fc_mean.values,
    "lower_ci": fc_ci.iloc[:, 0].values,
    "upper_ci": fc_ci.iloc[:, 1].values
}).round(2)

pred_2030 = float(forecast_df.loc[forecast_df["year"] == 2030, "forecast"].iloc[0])
lower_2030 = float(forecast_df.loc[forecast_df["year"] == 2030, "lower_ci"].iloc[0])
upper_2030 = float(forecast_df.loc[forecast_df["year"] == 2030, "upper_ci"].iloc[0])

# -------------------------------
# 6) PRINT RESULTS + SAVE CSV
# -------------------------------
print(f"Model: ARIMA{order}")
print(f"AIC: {res.aic:.2f} | BIC: {res.bic:.2f}")
print("\nForecast table for 2025–2030:")
print(forecast_df.to_string(index=False))
print(f"\nForecast for 2030: {pred_2030:.2f}  |  95% CI: [{lower_2030:.2f}, {upper_2030:.2f}]")

out_csv = OUT_DIR / "arima_023_forecast_2025_2030.csv"
forecast_df.to_csv(out_csv, index=False)

# -------------------------------
# 7) VISUALIZATION
# -------------------------------
plt.figure(figsize=(12, 5))

plt.plot(ts_series.index, ts_series.values, label="Actual (1993–2024)")
x_fc = pd.to_datetime(forecast_df["year"].astype(str) + "-12-31")
plt.plot(x_fc, forecast_df["forecast"].values, label="ARIMA(0,2,3) Forecast (2025–2030)")

plt.fill_between(
    x_fc,
    forecast_df["lower_ci"].values,
    forecast_df["upper_ci"].values,
    alpha=0.2,
    label="95% Confidence Interval"
)

plt.title("Global Piracy Incidents — ARIMA(0,2,3) Forecast to 2030")
plt.xlabel("Year")
plt.ylabel("Number of Incidents")
plt.grid(True, linewidth=0.3)
plt.legend()

out_png = OUT_DIR / "arima_023_forecast_2025_2030.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()

print("\nFiles saved:")
print(" -", out_csv)
print(" -", out_png)
