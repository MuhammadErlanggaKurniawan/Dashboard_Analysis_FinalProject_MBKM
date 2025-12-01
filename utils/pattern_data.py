# utils/pattern_data.py
import pandas as pd
from typing import List, Tuple
from shapely import wkt  # Pastikan ini ada
# ============================
# Konstanta
# ============================

RADAR_FEATURES: List[str] = [
    "ACTIVE_RATIO",
    "AKTIF_PER_10K",
    "KARYAWAN_PER_KOP_AKTIF",
    "MANAGER_PER_KOP_AKTIF",
    "PROP_BESAR",
    "PROP_KECIL",
    "PROP_MENENGAH",
    "PROP_MIKRO",
]

# Label cluster sesuai interpretasi kamu
CLUSTER_LABELS = {
    0: "Cluster 0 – Mikro-Intensif",
    1: "Cluster 1 – Struktural-Besar",
}


# ============================
# Helper kecil
# ============================

def _read_csv_smart(path: str) -> pd.DataFrame:
    """
    Baca CSV dengan deteksi delimiter sederhana (',' atau ';').
    Biar aman kalau file sempat ke-save pakai titik koma.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()

    # kalau lebih banyak ';' → pakai ';', else default ','
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    return df


def _drop_index_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buang kolom pertama kalau cuma index auto (Unnamed: 0, field_1, dll).
    """
    if not len(df.columns):
        return df

    first = df.columns[0]
    if "unnamed" in first.lower() or first.lower().startswith("field_") or first.strip() == "":
        return df.drop(columns=[first])
    return df


def _parse_numeric_series(s: pd.Series) -> pd.Series:
    """
    Bersihkan angka yang mungkin pakai format Indonesia:
    - koma sebagai desimal (0,123)
    - spasi
    lalu convert ke float. Non-angka jadi NaN.
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ============================
# Loader dasar
# ============================

def load_cluster_base() -> pd.DataFrame:
    """
    Load hasil_cluster_terbaik.csv yang berisi:
    KABKOT, fitur, Cluster, Silhouette, dll.
    Robus ke delimiter ',' / ';' dan kolom index.
    """
    path = "data/hasil_cluster_terbaik.csv"
    df = _read_csv_smart(path)
    df = _drop_index_like(df)

    # Normalisasi nama kolom 'Cluster'
    cluster_col = None
    for c in df.columns:
        if c.strip().lower() == "cluster":
            cluster_col = c
            break
    if cluster_col and cluster_col != "Cluster":
        df = df.rename(columns={cluster_col: "Cluster"})

    # pastikan Cluster numerik (Int64 biar boleh NaN)
    if "Cluster" in df.columns:
        df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").astype("Int64")

    print("=== DEBUG load_cluster_base ===")
    print("kolom:", list(df.columns))
    print("n_rows:", len(df))
    return df


def load_geo_table_and_geojson() -> pd.DataFrame:
    """
    Load hasil_clustering.csv:
    - kolom minimal: kabkot, Cluster
    - kalau belum ada lon/lat tapi ada 'geometry' (WKT), kita hitung centroid
    """
    path = "data/hasil_clustering.csv"
    gdf = _read_csv_smart(path)
    gdf = _drop_index_like(gdf)

    # Normalisasi nama kolom penting
    rename_map = {}
    for col in gdf.columns:
        key = col.strip().lower()
        if key == "kabkot":
            rename_map[col] = "kabkot"
        elif key == "cluster":
            rename_map[col] = "Cluster"
        elif key == "silhouette":
            rename_map[col] = "Silhouette"
        elif key in ("lon", "longitude"):
            rename_map[col] = "lon"
        elif key in ("lat", "latitude"):
            rename_map[col] = "lat"
    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    # ======================================================
    # 1) Kalau ada geometry tapi belum ada lon/lat → hitung centroid
    # ======================================================
    if "geometry" in gdf.columns and ("lon" not in gdf.columns or "lat" not in gdf.columns):
        try:
            geoms = gdf["geometry"].astype(str).apply(wkt.loads)
            gdf["lon"] = geoms.apply(lambda g: g.centroid.x)
            gdf["lat"] = geoms.apply(lambda g: g.centroid.y)

            print("lon range:", float(gdf["lon"].min()), "→", float(gdf["lon"].max()))
            print("lat range:", float(gdf["lat"].min()), "→", float(gdf["lat"].max()))
        except Exception as e:
            print("[WARN] gagal hitung centroid dari geometry:", e)

    # ======================================================
    # 2) Pastikan lon/lat numerik
    # ======================================================
    if "lon" in gdf.columns:
        gdf["lon"] = _parse_numeric_series(gdf["lon"])
    if "lat" in gdf.columns:
        gdf["lat"] = _parse_numeric_series(gdf["lat"])

    # Cluster → numerik
    if "Cluster" in gdf.columns:
        gdf["Cluster"] = pd.to_numeric(gdf["Cluster"], errors="coerce").astype("Int64")

    print("=== DEBUG load_geo_table_and_geojson ===")
    print("kolom:", list(gdf.columns))
    print("n_rows:", len(gdf))
    return gdf



# ============================
# Sumber untuk radar & silhouette
# ============================

def build_radar_source() -> pd.DataFrame:
    """
    Sumber data untuk radar chart:
      - setiap fitur di-Robust-minmax (0–1) setelah dibersihkan ke numerik
      - dihitung rata-rata per Cluster
      - di-unpivot jadi (Cluster, feature, value)

    Kalau tidak ada fitur valid → return DataFrame kosong
    (supaya Dash nggak nge-crash).
    """
    base = load_cluster_base().copy()

    if "Cluster" not in base.columns:
        print("[WARN] Kolom 'Cluster' tidak ditemukan di hasil_cluster_terbaik.csv")
        return pd.DataFrame(columns=["Cluster", "feature", "value"])

    scaled = base.copy()

    used_features: List[str] = []
    for col in RADAR_FEATURES:
        if col not in scaled.columns:
            print(f"[WARN] Kolom fitur '{col}' tidak ada di dataset, di-skip.")
            continue

        # paksa ke numerik (handle format Indonesia)
        scaled[col] = _parse_numeric_series(scaled[col])

        col_min = scaled[col].min(skipna=True)
        col_max = scaled[col].max(skipna=True)

        if pd.isna(col_min) or pd.isna(col_max):
            print(f"[WARN] Kolom fitur '{col}' semuanya NaN / non-numerik, di-skip.")
            continue

        if col_max > col_min:
            scaled[col] = (scaled[col] - col_min) / (col_max - col_min)
        else:
            scaled[col] = 0.0

        used_features.append(col)

    if not used_features:
        print("[WARN] Tidak ada fitur yang bisa dipakai untuk radar chart.")
        return pd.DataFrame(columns=["Cluster", "feature", "value"])

    grouped = (
        scaled.groupby("Cluster")[used_features]
        .mean(numeric_only=True)
        .reset_index()
    )

    rows = []
    for _, row in grouped.iterrows():
        cl = int(row["Cluster"])
        for f in used_features:
            rows.append(
                {
                    "Cluster": cl,
                    "feature": f,
                    "value": float(row[f]),
                }
            )

    radar_df = pd.DataFrame(rows)
    return radar_df


def build_silhouette_source() -> pd.DataFrame:
    """
    Sumber untuk bar chart silhouette:
      - satu bar per kab/kota
    """
    base = load_cluster_base().copy()
    cols = ["KABKOT", "Cluster", "Silhouette"]
    cols = [c for c in cols if c in base.columns]
    if not cols:
        return pd.DataFrame(columns=["KABKOT", "Cluster", "Silhouette"])

    sil = base[cols].copy()

    # pastikan numerik
    if "Silhouette" in sil.columns:
        sil["Silhouette"] = _parse_numeric_series(sil["Silhouette"])

    return sil


def compute_overall_silhouette() -> float:
    """
    Rata-rata silhouette global dari hasil_cluster_terbaik.csv
    """
    base = load_cluster_base()
    if "Silhouette" not in base.columns:
        return float("nan")

    s = _parse_numeric_series(base["Silhouette"])
    s = s.dropna()
    if s.empty:
        return float("nan")

    return float(s.mean())
