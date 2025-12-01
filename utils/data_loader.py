# utils/data_loader.py
import pandas as pd
import numpy as np

# =======================
# TSA (Time Series)
# =======================

def load_tsa_data():
    """
    Load data TSA dari CSV files.
    - data/tsa_historis.csv      -> kolom: periode (YYYY-MM), jumlah_penumpang
    - data/tsa_forecast_2025.csv -> kolom: Bulan, Forecast, Lower_95%, Upper_95%, Lower_80%, Upper_80%
    """
    try:
        # --- data historis ---
        hist_df = pd.read_csv('data/tsa_historis.csv')
        hist_df['periode'] = pd.to_datetime(hist_df['periode'])
        hist_df = hist_df.rename(columns={'jumlah_penumpang': 'actual'})
        hist_df['actual'] = hist_df['actual'].astype(float)
        hist_df = hist_df.sort_values('periode')

        # --- data forecast ---
        forecast_df = pd.read_csv('data/tsa_forecast_2025.csv')
        forecast_df['Bulan'] = pd.to_datetime(forecast_df['Bulan'])
        forecast_df = forecast_df.rename(columns={
            'Forecast': 'forecast',
            'Lower_95%': 'lower_95',
            'Upper_95%': 'upper_95',
            'Lower_80%': 'lower_80',
            'Upper_80%': 'upper_80'
        })
        for col in ['forecast', 'lower_95', 'upper_95', 'lower_80', 'upper_80']:
            forecast_df[col] = forecast_df[col].astype(float)
        forecast_df = forecast_df.sort_values('Bulan')

        return hist_df, forecast_df

    except Exception as e:
        print(f"❌ Error loading TSA data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_tsa_metrics(hist_df, forecast_df):
    """
    Hitung KPI summary untuk kartu di TSA tab.
    """
    if hist_df.empty:
        return {}

    try:
        avg_passengers = hist_df['actual'].mean()
        peak_passengers = hist_df['actual'].max()
        lowest_passengers = hist_df['actual'].min()

        peak_data = hist_df[hist_df['actual'] == peak_passengers].iloc[0]
        peak_month = peak_data['periode'].strftime('%b %Y')

        lowest_data = hist_df[hist_df['actual'] == lowest_passengers].iloc[0]
        lowest_month = lowest_data['periode'].strftime('%b %Y')

        data_2023 = hist_df[hist_df['periode'].dt.year == 2023]['actual'].mean()
        data_2024 = hist_df[hist_df['periode'].dt.year == 2024]['actual'].mean()
        growth_rate = ((data_2024 - data_2023) / data_2023) * 100 if data_2023 > 0 else 0

        return {
            'avg_passengers': avg_passengers,
            'peak_passengers': peak_passengers,
            'peak_month': peak_month,
            'lowest_passengers': lowest_passengers,
            'lowest_month': lowest_month,
            'growth_rate': growth_rate
        }
    except Exception as e:
        print(f"❌ Error calculating TSA metrics: {e}")
        return {}

# =======================
# NONPARAMETRIK KOPERASI
# =======================

def load_cooperative_data():
    """
    Load dataset koperasi asli (All Data) dari CSV.
    File ini hasil export sheet 'All Data' dari:
    Dataset MBKM (A) (3).xlsx
    """
    path = 'data/cooperative_raw_all_data.csv'
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded cooperative_raw_all_data.csv: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading cooperative data from {path}: {e}")
        return pd.DataFrame()
