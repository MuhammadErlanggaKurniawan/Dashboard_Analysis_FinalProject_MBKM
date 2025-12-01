# utils/cooperative_processor.py
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Daftar variabel numerik utama (tanpa rasio turunan)
NUMERIC_COLS = [
    'total_penduduk',
    'jumlah_koperasi_aktif',
    'jumlah_koperasi_tidak_aktif',
    'jumlah_koperasi_total',
    'jumlah_karyawan',
    'jumlah_manager',
    'usaha_besar',
    'usaha_kecil',
    'usaha_menengah',
    'usaha_mikro',
]


def preprocess_cooperative_data(df):
    """
    Preprocessing data koperasi untuk analisis nonparametrik.
    - Rapikan nama kolom
    - Buat total_penduduk
    - Buat jenis_wilayah
    - Pastikan kolom numerik dalam bentuk numeric
    """
    df_processed = df.copy()

    # Kadang kolom nama masih "KABUPATEN / KOTA" → standarkan dulu
    df_processed = df_processed.rename(columns={"KABUPATEN / KOTA": "KABUPATEN KOTA"})

    # Rapikan kolom: lower, underscore
    df_processed.columns = (
        df_processed.columns
        .str.strip()
        .str.lower()
        .str.replace(" / ", "_")
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Buat total_penduduk
    if "jumlah_penduduk_laki_laki" in df_processed.columns and \
       "jumlah_penduduk_perempuan" in df_processed.columns:
        df_processed["total_penduduk"] = (
            df_processed["jumlah_penduduk_laki_laki"] +
            df_processed["jumlah_penduduk_perempuan"]
        )

    # Buat jenis_wilayah (kota / kabupaten)
    if "kabupaten_kota" in df_processed.columns:
        df_processed["jenis_wilayah"] = np.where(
            df_processed["kabupaten_kota"].str.contains("Kota", case=False, na=False),
            "Kota", "Kabupaten"
        )

    # Konversi kolom numerik
    for col in NUMERIC_COLS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    return df_processed


def get_available_periods(df):
    """
    Ambil daftar periode_update unik untuk dropdown (misal 2018-Q4, 2019-Q1 dst).
    """
    if "periode_update" not in df.columns:
        return []

    periods = (
        df["periode_update"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    periods = sorted(periods)
    return periods


def calculate_spearman_correlations(df):
    """
    Hitung korelasi Spearman + matriks p-value untuk variabel numerik utama.
    """
    corr_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if len(corr_cols) < 2:
        empty = pd.DataFrame()
        return empty, empty

    sub_df = df[corr_cols].copy().dropna()
    if sub_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    corr_matrix = sub_df.corr(method='spearman')
    p_values = calculate_spearman_pvalues(sub_df)

    return corr_matrix, p_values


def calculate_spearman_pvalues(df_numeric):
    """
    Hitung p-values untuk korelasi Spearman.
    """
    df = df_numeric.dropna()
    n_vars = df.shape[1]
    p_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                corr, p_value = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
                p_matrix[i, j] = p_value
            else:
                p_matrix[i, j] = 0.0

    return pd.DataFrame(p_matrix, index=df.columns, columns=df.columns)


def perform_mannwhitney_test(df, variable):
    """
    Uji Mann-Whitney untuk perbandingan Kota vs Kabupaten pada satu variabel.
    """
    if "jenis_wilayah" not in df.columns or variable not in df.columns:
        return None

    kota_data = df[df['jenis_wilayah'] == 'Kota'][variable].dropna()
    kab_data = df[df['jenis_wilayah'] == 'Kabupaten'][variable].dropna()

    if len(kota_data) < 2 or len(kab_data) < 2:
        return None

    stat, p_value = stats.mannwhitneyu(kota_data, kab_data, alternative='two-sided')
    cles = stat / (len(kota_data) * len(kab_data))

    interpretation = 'Kota > Kabupaten' if cles > 0.5 else 'Kabupaten > Kota'

    return {
        'statistic': stat,
        'p_value': p_value,
        'effect_size': cles,
        'interpretation': interpretation,
        'kota_median': kota_data.median(),
        'kabupaten_median': kab_data.median()
    }


def get_top_regions(df, variable, n=10):
    """
    Ambil top n wilayah berdasarkan variabel tertentu.
    """
    if variable not in df.columns:
        return pd.DataFrame()

    cols = ['kabupaten_kota', 'jenis_wilayah', variable]
    available = [c for c in cols if c in df.columns]
    return df[available].nlargest(n, variable)


def get_statistical_insights(df):
    """
    Generate beberapa insight singkat untuk cards di dashboard.
    """
    insights = []

    # 1) Korelasi terkuat antar variabel numerik
    corr_matrix, _ = calculate_spearman_correlations(df)
    if not corr_matrix.empty:
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                rho = corr_matrix.iloc[i, j]
                strong_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    rho
                ))
        if strong_pairs:
            strongest = max(strong_pairs, key=lambda x: abs(x[2]))
            insights.append(
                f"📈 Korelasi terkuat: {strongest[0]} vs {strongest[1]} (ρ={strongest[2]:.3f})"
            )

    # 2) Perbedaan Kota vs Kabupaten untuk jumlah_koperasi_aktif
    mw_test = perform_mannwhitney_test(df, 'jumlah_koperasi_aktif')
    if mw_test and mw_test['p_value'] < 0.05:
        insights.append(
            f"🏙️ {mw_test['interpretation']} dalam jumlah koperasi aktif (p={mw_test['p_value']:.3f})"
        )

    # 3) Wilayah dengan koperasi aktif terbanyak
    top_koperasi = get_top_regions(df, 'jumlah_koperasi_aktif', 1)
    if not top_koperasi.empty:
        insights.append(
            f"🏆 {top_koperasi.iloc[0]['kabupaten_kota']} memiliki koperasi aktif terbanyak"
        )

    # 4) Disparitas usaha mikro
    if 'usaha_mikro' in df.columns:
        max_umkm = df['usaha_mikro'].max()
        min_umkm = df['usaha_mikro'].min()
        if pd.notna(max_umkm) and pd.notna(min_umkm) and min_umkm > 0:
            disparity = max_umkm / min_umkm
            insights.append(
                f"⚡ Disparitas usaha mikro: {disparity:.1f}x antara wilayah tertinggi dan terendah"
            )

    return insights
