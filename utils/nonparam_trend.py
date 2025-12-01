# utils/nonparam_trend.py

import pandas as pd
import numpy as np
from scipy import stats

# pakai daftar variabel numerik yang sama dengan pipeline nonparametrik
from utils.cooperative_processor import NUMERIC_COLS

# Label cantik untuk semua variabel numerik utama
VAR_LABELS = {
    "jumlah_koperasi_aktif": "Jumlah Koperasi Aktif",
    "jumlah_koperasi_tidak_aktif": "Jumlah Koperasi Tidak Aktif",
    "jumlah_koperasi_total": "Jumlah Koperasi Total",
    "jumlah_karyawan": "Jumlah Karyawan Koperasi",
    "jumlah_manager": "Jumlah Manager Koperasi",
    "usaha_besar": "Koperasi Usaha Besar",
    "usaha_kecil": "Koperasi Usaha Kecil",
    "usaha_menengah": "Koperasi Usaha Menengah",
    "usaha_mikro": "Koperasi Usaha Mikro",
    "total_penduduk": "Total Penduduk",
}


def _load_nonparam_base(path: str = "data/cooperative_raw_all_data.csv") -> pd.DataFrame:
    """
    Load data mentah koperasi (semua periode).
    Normalisasi nama kolom biar konsisten dengan pipeline nonparametrik.
    """
    df = pd.read_csv(path)

    # normalisasi nama kolom
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" / ", "_")
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # handle variasi nama kabupaten/kota
    if "kabupaten_/_kota" in df.columns:
        df = df.rename(columns={"kabupaten_/_kota": "kabupaten_kota"})
    if "kabupaten_kota" not in df.columns:
        raise ValueError("Kolom 'kabupaten_kota' tidak ditemukan di cooperative_raw_all_data.csv")

    if "periode_update" not in df.columns:
        raise ValueError("Kolom 'periode_update' tidak ditemukan di cooperative_raw_all_data.csv")

    # pastikan kolom penduduk ada
    if "jumlah_penduduk_laki_laki" in df.columns and "jumlah_penduduk_perempuan" in df.columns:
        df["total_penduduk"] = (
            df["jumlah_penduduk_laki_laki"].fillna(0)
            + df["jumlah_penduduk_perempuan"].fillna(0)
        )

    # jenis_wilayah: Kota vs Kabupaten
    df["jenis_wilayah"] = np.where(
        df["kabupaten_kota"].str.contains("kota", case=False, na=False),
        "Kota",
        "Kabupaten",
    )

    return df


def _kategori_r(r_abs: float) -> str:
    """Kategori kekuatan effect size r."""
    if r_abs < 0.10:
        return "sangat kecil"
    elif r_abs < 0.30:
        return "kecil"
    elif r_abs < 0.50:
        return "sedang"
    else:
        return "besar"


def compute_mannwhitney_trend() -> pd.DataFrame:
    """
    Hitung tren effect size r Mann–Whitney (Kota vs Kabupaten) per periode
    untuk SEMUA variabel numerik utama (NUMERIC_COLS).

    Return: DataFrame dengan kolom:
      ['periode', 'variabel', 'label_variabel', 'U', 'p_value',
       'CLES', 'z', 'r', 'r_abs', 'kategori_r', 'signif',
       'n_kota', 'n_kab']
    """
    df = _load_nonparam_base()

    # urutkan periode secara kronologis (string "YYYY-Qx")
    all_periods = sorted(df["periode_update"].dropna().unique().tolist())

    # filter hanya periode mulai 2019 (permintaan dosen)
    periods = [p for p in all_periods if str(p)[:4] >= "2019"]

    results = []

    for per in periods:
        sub = df[df["periode_update"] == per].copy()
        if sub.empty:
            continue

        # pakai semua variabel numerik yang kamu pakai di analisis nonparametrik
        for var in NUMERIC_COLS:
            if var not in sub.columns:
                continue

            kota = sub[sub["jenis_wilayah"] == "Kota"][var].dropna()
            kab = sub[sub["jenis_wilayah"] == "Kabupaten"][var].dropna()

            n1, n2 = len(kota), len(kab)
            if n1 < 2 or n2 < 2:
                continue

            stat, p_val = stats.mannwhitneyu(kota, kab, alternative="two-sided")
            cles = stat / (n1 * n2)

            # hitung z dan r (approx)
            mean_U = n1 * n2 / 2
            sd_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

            if sd_U > 0:
                z = (stat - mean_U) / sd_U
                r = z / np.sqrt(n1 + n2)
            else:
                z = np.nan
                r = np.nan

            r_abs = float(np.abs(r)) if not np.isnan(r) else np.nan
            kategori = _kategori_r(r_abs) if not np.isnan(r_abs) else "-"
            signif = "SIGNIFIKAN" if p_val < 0.05 else "Tidak signifikan"

            results.append(
                {
                    "periode": per,
                    "variabel": var,
                    "label_variabel": VAR_LABELS.get(var, var),
                    "U": float(stat),
                    "p_value": float(p_val),
                    "CLES": float(cles),
                    "z": float(z) if not np.isnan(z) else np.nan,
                    "r": float(r) if not np.isnan(r) else np.nan,
                    "r_abs": r_abs,
                    "kategori_r": kategori,
                    "signif": signif,
                    "n_kota": int(n1),
                    "n_kab": int(n2),
                }
            )

    trend_df = pd.DataFrame(results)

    # sort kronologis: parse "YYYY-Qx"
    def _periode_key(s):
        s = str(s)
        year = int(s[:4])
        # ambil angka terakhir sebagai kuartal, kalau ada
        try:
            q = int(s[-1])
        except ValueError:
            q = 4
        return year * 10 + q

    if not trend_df.empty:
        trend_df = trend_df.sort_values(
            by="periode",
            key=lambda col: col.map(_periode_key)
        )

    print("✅ compute_mannwhitney_trend() – shape:", trend_df.shape)
    return trend_df
