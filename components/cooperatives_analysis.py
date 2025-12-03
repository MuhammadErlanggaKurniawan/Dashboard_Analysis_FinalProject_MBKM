# components/cooperatives_analysis.py

import dash # type: ignore
from dash import dcc, html, Input, Output, callback, dash_table # type: ignore
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats

from utils.nonparam_trend import compute_mannwhitney_trend, VAR_LABELS
from utils.data_loader import load_cooperative_data
from utils.cooperative_processor import (
    preprocess_cooperative_data,
    calculate_spearman_correlations,
    perform_mannwhitney_test,
    get_top_regions,
    get_statistical_insights,
    get_available_periods,
    NUMERIC_COLS,
)

VAR_NAME_MAP = {
    "total_penduduk": "Total Penduduk",
    "jumlah_koperasi_aktif": "Jumlah Koperasi Aktif",
    "jumlah_koperasi_tidak_aktif": "Koperasi Tidak Aktif",
    "jumlah_koperasi_total": "Total Koperasi",
    "jumlah_karyawan": "Jumlah Karyawan",
    "jumlah_manager": "Jumlah Manajer",
    "usaha_besar": "Usaha Besar",
    "usaha_kecil": "Usaha Kecil",
    "usaha_menengah": "Usaha Menengah",
    "usaha_mikro": "Usaha Mikro",
}


# ==========================================
# 0. GLOBAL DATA
# ==========================================

# Tren effect size Mann–Whitney (Kota vs Kabupaten) lintas periode
_mw_trend_df = compute_mannwhitney_trend()

# Data utama koperasi
_raw_df = load_cooperative_data()
_df_processed_global = preprocess_cooperative_data(_raw_df)
PERIODE_LIST = get_available_periods(_df_processed_global)

# Opsi variabel untuk dropdown (pakai variabel real)
CORR_VARIABLE_OPTIONS = [
    {"label": "Jumlah Koperasi Aktif", "value": "jumlah_koperasi_aktif"},
    {"label": "Jumlah Koperasi Tidak Aktif", "value": "jumlah_koperasi_tidak_aktif"},
    {"label": "Jumlah Koperasi Total", "value": "jumlah_koperasi_total"},
    {"label": "Jumlah Karyawan", "value": "jumlah_karyawan"},
    {"label": "Jumlah Manager", "value": "jumlah_manager"},
    {"label": "Usaha Besar", "value": "usaha_besar"},
    {"label": "Usaha Kecil", "value": "usaha_kecil"},
    {"label": "Usaha Menengah", "value": "usaha_menengah"},
    {"label": "Usaha Mikro", "value": "usaha_mikro"},
    {"label": "Total Penduduk", "value": "total_penduduk"},
]


# ==========================================
# LAYOUT UTAMA
# ==========================================

def create_cooperatives_layout():
    """Layout utama untuk analisis koperasi nonparametrik"""
    default_periode = PERIODE_LIST[-1] if PERIODE_LIST else None

    # --- Section tren lintas periode (baru) ---
    trend_section = html.Div(
        [
            html.H3(
                "📈 Ringkasan Lintas Periode – Kesenjangan Kota vs Kabupaten (2019–2025)",
                className="mt-4 mb-3",
            ),
            # Baris atas: dropdown pilih variabel
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Pilih Variabel Kunci", className="fw-bold"),
                            dcc.Dropdown(
                                id="coop-trend-variable",
                                options=[
                                    {
                                        "label": VAR_LABELS.get(v, v),
                                        "value": v,
                                    }
                                    for v in (
                                        _mw_trend_df["variabel"].unique().tolist()
                                        if not _mw_trend_df.empty
                                        else []
                                    )
                                ],
                                value=(
                                    _mw_trend_df["variabel"].unique()[0]
                                    if not _mw_trend_df.empty
                                    else None
                                ),
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        className="col-md-4",
                    ),
                ],
                className="row",
            ),
            # Baris bawah: grafik kiri, card besar kanan
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="coop-trend-effect-graph",
                                style={"height": "420px"},
                            )
                        ],
                        className="col-md-8",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        "📝 Kesimpulan & Analisis Kebijakan",
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        id="coop-trend-summary-text",
                                        className="small",
                                    ),
                                    html.Hr(),
                                    html.P(
                                        "r di sini adalah ukuran efek (effect size) dari uji Mann–Whitney "
                                        "untuk membandingkan Kota vs Kabupaten. Semakin besar |r|, "
                                        "semakin kuat perbedaan distribusi antara keduanya.",
                                        className="text-muted",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                    html.P(
                                        "Rule of thumb: |r| < 0.10 sangat kecil, 0.10–0.30 kecil, "
                                        "0.30–0.50 sedang, > 0.50 besar.",
                                        className="text-muted",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ],
                                className="card p-3 shadow-sm position-sticky",
                                style={"top": "80px", "maxHeight": "420px", "overflowY": "auto"},
                            ),
                        ],
                        className="col-md-4",
                    ),
                ],
                className="row",
            ),
        ],
        className="card p-3 shadow-sm mt-4",
    )

    return html.Div(
        [
            # HEADER
            html.Div(
                [
                    html.H1(
                        "📊 Analisis Nonparametrik - Koperasi & UMKM Jawa Timur",
                        className="text-center mb-2 text-primary",
                    ),
                    html.P(
                        "Analisis korelasi dan perbedaan wilayah (Kota vs Kabupaten) "
                        "berbasis data koperasi, UMKM, dan penduduk Jawa Timur.",
                        className="text-center text-muted mb-4",
                    ),
                ],
                className="header-section",
            ),

            # FILTER CONTROLS (Periode, Variabel, Wilayah, Metode)
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("🗓️ Pilih Periode Data:", className="fw-bold"),
                            dcc.Dropdown(
                                id="periode-selector",
                                options=[{"label": p, "value": p} for p in PERIODE_LIST],
                                value=default_periode,
                                placeholder="Pilih periode (mis. 2018-Q4)",
                                className="mb-3",
                            ),
                        ],
                        className="col-md-3",
                    ),
                    html.Div(
                        [
                            html.Label("🎯 Variabel Utama:", className="fw-bold"),
                            dcc.Dropdown(
                                id="variable-selector",
                                options=CORR_VARIABLE_OPTIONS,
                                value="jumlah_koperasi_aktif",
                                className="mb-3",
                            ),
                        ],
                        className="col-md-3",
                    ),
                    html.Div(
                        [
                            html.Label("🏙️ Filter Wilayah:", className="fw-bold"),
                            dcc.RadioItems(
                                id="region-type",
                                options=[
                                    {"label": " Semua Wilayah", "value": "all"},
                                    {"label": " Kota Saja", "value": "Kota"},
                                    {"label": " Kabupaten Saja", "value": "Kabupaten"},
                                ],
                                value="all",
                                inline=True,
                                className="mt-2",
                            ),
                        ],
                        className="col-md-3",
                    ),
                    html.Div(
                        [
                            html.Label("📈 Metode Statistik:", className="fw-bold"),
                            dcc.Dropdown(
                                id="stat-method",
                                options=[
                                    {
                                        "label": "Korelasi Spearman",
                                        "value": "spearman",
                                    },
                                    {
                                        "label": "Uji Mann-Whitney (Kota vs Kabupaten)",
                                        "value": "mannwhitney",
                                    },
                                    {
                                        "label": "Analisis Top Wilayah",
                                        "value": "top-regions",
                                    },
                                ],
                                value="spearman",
                                className="mb-3",
                            ),
                        ],
                        className="col-md-3",
                    ),
                ],
                className="row mb-4 p-3 bg-light rounded shadow-sm",
            ),

            # INSIGHTS CARDS
            html.Div(id="insights-cards", className="row mb-4"),

            # MAIN CONTENT
            html.Div(
                [
                    # LEFT COLUMN
                    html.Div(
                        [
                            # SHAPIRO TABLE
                            html.Div(
                                [
                                    html.H4(
                                        "🧪 Uji Normalitas Shapiro–Wilk",
                                        className="card-title",
                                    ),
                                    html.P(
                                        "Menunjukkan apakah distribusi variabel numerik normal "
                                        "atau tidak pada periode terpilih.",
                                        className="text-muted",
                                    ),
                                    html.Div(id="shapiro-table"),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            # HEATMAP
                            html.Div(
                                [
                                    html.H4(
                                        "🔥 Heatmap Korelasi Spearman",
                                        className="card-title",
                                    ),
                                    html.P(
                                        "Korelasi rank Spearman antar variabel ekonomi "
                                        "(hanya variabel non-normal yang relevan dalam konteks nonparametrik).",
                                        className="text-muted",
                                    ),
                                    dcc.Graph(id="correlation-heatmap"),
                                    html.Div(
                                        id="correlation-interpretation",
                                        className="mt-2 p-2 bg-info text-white rounded small",
                                    ),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            # SCATTERPLOT
                            html.Div(
                                [
                                    html.H4(
                                        "📈 Scatterplot Korelasi Spearman",
                                        className="card-title",
                                    ),
                                    html.P(
                                        "Visualisasi hubungan antar dua variabel untuk periode "
                                        "dan filter wilayah yang dipilih.",
                                        className="text-muted",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Variabel X",
                                                        className="fw-bold small",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="scatter-x",
                                                        options=CORR_VARIABLE_OPTIONS,
                                                        value="jumlah_koperasi_aktif",
                                                        className="mb-2",
                                                    ),
                                                ],
                                                className="col-md-6",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Variabel Y",
                                                        className="fw-bold small",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="scatter-y",
                                                        options=CORR_VARIABLE_OPTIONS,
                                                        value="total_penduduk",
                                                        className="mb-2",
                                                    ),
                                                ],
                                                className="col-md-6",
                                            ),
                                        ],
                                        className="row",
                                    ),
                                    dcc.Graph(id="correlation-scatter"),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            # BOXPLOT
                            html.Div(
                                [
                                    html.H4(
                                        "📦 Perbandingan Distribusi Kota vs Kabupaten",
                                        className="card-title",
                                    ),
                                    html.P(
                                        "Boxplot perbandingan distribusi variabel utama antara "
                                        "kota dan kabupaten.",
                                        className="text-muted",
                                    ),
                                    dcc.Graph(id="distribution-boxplot"),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                        ],
                        className="col-md-8",
                    ),
                    # RIGHT COLUMN
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4(
                                        "📊 Hasil Uji Statistik", className="card-title"
                                    ),
                                    html.Div(
                                        id="statistical-results",
                                        className="stats-container",
                                    ),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            html.Div(
                                [
                                    html.H4("⭐ Highlight Periode Ini", className="card-title"),
                                    html.Div(id="period-highlights", className="small"),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            html.Div(
                                [
                                    html.H4(
                                        "📈 Tren Singkat Variabel Utama", className="card-title"
                                    ),
                                    dcc.Graph(
                                        id="mini-trend-main-var",
                                        style={"height": "190px"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            html.Div(
                                [
                                    html.H4(
                                        "📊 Komposisi Jenis Usaha", className="card-title"
                                    ),
                                    dcc.Graph(
                                        id="mini-jenis-usaha",
                                        style={"height": "220px"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            html.Div(
                                [
                                    html.H4(
                                        "🏆 Top 5 Wilayah", className="card-title"
                                    ),
                                    html.Div(
                                        id="top-regions-table",
                                        className="regions-table",
                                    ),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            html.Div(
                                [
                                    html.H4(
                                        "🧭 Kesimpulan & Analisis Kebijakan",
                                        className="card-title",
                                    ),
                                    html.Div(
                                        id="policy-summary",
                                        className="small",
                                    ),
                                ],
                                className="card p-3 mb-3 shadow-sm",
                            ),
                            html.Div(
                                [
                                    html.H4(
                                        "ℹ️ Info Dataset", className="card-title"
                                    ),
                                    html.Div(
                                        id="dataset-info", className="dataset-info"
                                    ),
                                ],
                                className="card p-3 shadow-sm",
                            ),
                        ],
                        className="col-md-4",
                    ),
                ],
                className="row",
            ),

            # DETAILED ANALYSIS SECTION
            html.Div(
                [
                    html.H3("🔍 Analisis Detail Regional", className="mb-3"),
                    dcc.Graph(id="regional-analysis"),
                ],
                className="mt-4 card p-3 shadow-sm",
            ),

            # SECTION TREN LINTAS PERIODE (BARU)
            trend_section,
        ],
        className="container-fluid",
    )


# =========================================================
# CALLBACK UTAMA (PER PERIODE)
# =========================================================

@callback(
    [
        Output("correlation-heatmap", "figure"),
        Output("distribution-boxplot", "figure"),
        Output("statistical-results", "children"),
        Output("top-regions-table", "children"),
        Output("regional-analysis", "figure"),
        Output("insights-cards", "children"),
        Output("dataset-info", "children"),
        Output("correlation-interpretation", "children"),
        Output("correlation-scatter", "figure"),
        Output("shapiro-table", "children"),
        Output("policy-summary", "children"),
        Output("period-highlights", "children"),
        Output("mini-trend-main-var", "figure"),
        Output("mini-jenis-usaha", "figure"),
    ],
    [
        Input("variable-selector", "value"),
        Input("region-type", "value"),
        Input("stat-method", "value"),
        Input("scatter-x", "value"),
        Input("scatter-y", "value"),
        Input("periode-selector", "value"),
    ],
)

def update_analysis(
    selected_variable,
    region_type,
    stat_method,
    scatter_x,
    scatter_y,
    selected_periode,
):

    # === 1. Load & preprocess ===
    df = load_cooperative_data()
    df_processed = preprocess_cooperative_data(df)

    # Filter periode dulu
    if selected_periode and "periode_update" in df_processed.columns:
        df_processed = df_processed[df_processed["periode_update"] == selected_periode]

    # Filter jenis wilayah
    if region_type != "all" and "jenis_wilayah" in df_processed.columns:
        df_processed = df_processed[df_processed["jenis_wilayah"] == region_type]

    # Safety check
    if df_processed.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Data kosong untuk kombinasi filter ini.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        empty_fig.update_layout(height=400)
        msg = "Data kosong untuk periode / filter yang dipilih."

        return (
            empty_fig,          # correlation-heatmap
            empty_fig,          # distribution-boxplot
            html.Div(msg),      # statistical-results
            html.Div(msg),      # top-regions-table
            empty_fig,          # regional-analysis
            [],                 # insights-cards
            html.Div(msg),      # dataset-info
            msg,                # correlation-interpretation
            empty_fig,          # correlation-scatter
            html.Div(msg),      # shapiro-table
            html.Div(msg),      # policy-summary
            html.Div(msg),      # period-highlights
            empty_fig,          # mini-trend-main-var
            empty_fig,          # mini-jenis-usaha
        )


    # === 2. Shapiro–Wilk table (per variabel numerik) ===
    shapiro_table = create_shapiro_table(df_processed)

    # === 3. Heatmap & interpretasi korelasi ===
    heatmap_fig, strongest_text = create_correlation_heatmap_strict(df_processed)

    # === 4. Scatterplot ===
    scatter_fig = create_correlation_scatter_strict(
        df_processed, scatter_x, scatter_y
    )

    # === 5. Komponen lain ===
    box_fig = create_distribution_boxplot(df_processed, selected_variable)
    stats_results = generate_statistical_results(
        df_processed, stat_method, selected_variable
    )
    top_regions_table = generate_top_regions_table(df_processed, selected_variable)
    regional_fig = create_regional_analysis(df_processed, selected_variable)
    insights_cards = create_insights_cards(df_processed, selected_periode)
    dataset_info = create_dataset_info(df_processed)

    corr_interpretation = strongest_text  # teks di bawah heatmap
    policy_summary = create_policy_summary(
        df_processed,
        selected_periode,
        region_type,
        selected_variable,
    )
    period_highlights = create_period_highlights(
        df_processed,
        selected_periode,
        region_type,
    )

    mini_trend_fig = create_mini_trend_main_var(
        _df_processed_global,
        selected_variable,
        region_type,
    )

    mini_jenis_usaha_fig = create_mini_jenis_usaha_figure(df_processed)

    return (
        heatmap_fig,          # correlation-heatmap
        box_fig,              # distribution-boxplot
        stats_results,        # statistical-results
        top_regions_table,    # top-regions-table
        regional_fig,         # regional-analysis
        insights_cards,       # insights-cards
        dataset_info,         # dataset-info
        corr_interpretation,  # correlation-interpretation
        scatter_fig,          # correlation-scatter
        shapiro_table,        # shapiro-table
        policy_summary,       # policy-summary
        period_highlights,    # period-highlights
        mini_trend_fig,       # mini-trend-main-var
        mini_jenis_usaha_fig, # mini-jenis-usaha
    )



# =========================================================
# CALLBACK TREN LINTAS PERIODE (BARU)
# =========================================================

@callback(
    Output("coop-trend-effect-graph", "figure"),
    Input("coop-trend-variable", "value"),
)
def update_coop_trend_graph(var_name):
    if _mw_trend_df.empty or var_name is None:
        return go.Figure()

    df = _mw_trend_df[_mw_trend_df["variabel"] == var_name].copy()
    if df.empty:
        return go.Figure()

    x = df["periode"].tolist()
    y = df["r_abs"].tolist()
    signif_flags = df["signif"].tolist()

    marker_colors = [
        "#d62728" if s == "SIGNIFIKAN" else "#7f7f7f" for s in signif_flags
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="|r| (kekuatan efek)",
            marker=dict(size=8, color=marker_colors),
            line=dict(width=2),
            hovertemplate="<b>%{x}</b><br>|r| = %{y:.3f}<extra></extra>",
        )
    )

    # Garis bantu kategori r
    for thr, label in [(0.1, "kecil"), (0.3, "sedang"), (0.5, "besar")]:
        fig.add_hline(
            y=thr,
            line_dash="dot",
            line_color="lightgray",
            annotation_text=label,
            annotation_position="right",
        )

    ymax = max(y) if len(y) else 0.0
    fig.update_layout(
        title=f"Tren Effect Size Mann–Whitney (Kota vs Kabupaten) – {VAR_LABELS.get(var_name, var_name)}",
        xaxis_title="Periode",
        yaxis_title="|r| (kekuatan efek)",
        yaxis=dict(range=[0, max(0.6, ymax * 1.1)]),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


@callback(
    Output("coop-trend-summary-text", "children"),
    Input("coop-trend-variable", "value"),
)
def update_coop_trend_summary(var_name):
    if _mw_trend_df.empty or var_name is None:
        return "Data tren tidak tersedia."

    df = _mw_trend_df[_mw_trend_df["variabel"] == var_name].copy()
    if df.empty:
        return "Data tren tidak tersedia."

    var_label = VAR_LABELS.get(var_name, var_name)

    # statistik dasar
    r_vals = df["r_abs"].dropna()
    if r_vals.empty:
        return "Effect size r tidak tersedia."

    first = df.iloc[0]
    last = df.iloc[-1]

    r0 = float(first["r_abs"])
    rT = float(last["r_abs"])
    k0 = first["kategori_r"]
    kT = last["kategori_r"]

    periode_awal = str(first["periode"])
    periode_akhir = str(last["periode"])

    n_total = len(df)
    n_signif = int((df["signif"] == "SIGNIFIKAN").sum())
    r_min = float(r_vals.min())
    r_max = float(r_vals.max())
    r_mean = float(r_vals.mean())

    # arah tren kasar
    delta = rT - r0
    if delta > 0.05:
        arah_tren = "cenderung menguat"
    elif delta < -0.05:
        arah_tren = "cenderung melemah"
    else:
        arah_tren = "relatif stabil"

    # narasi kebijakan simple per variabel
    if var_name == "jumlah_koperasi_aktif":
        policy_points = [
            "Kabupaten secara konsisten memiliki jumlah koperasi aktif yang lebih kuat dibanding Kota.",
            "Wilayah Kota butuh dorongan pembentukan dan penguatan koperasi baru (insentif, pendampingan, integrasi dengan UMKM).",
            "Di Kabupaten fokus diarahkan ke peningkatan kualitas dan produktivitas koperasi yang sudah banyak berdiri.",
        ]
    elif var_name == "usaha_mikro":
        policy_points = [
            "Fluktuasi kekuatan efek menunjukkan dinamika yang cukup besar pada basis koperasi usaha mikro.",
            "Periode dengan |r| besar bisa dibaca sebagai momentum keberhasilan program tertentu yang layak direplikasi.",
            "Penguatan pendampingan manajemen, akses pembiayaan, dan digitalisasi layanan koperasi mikro menjadi prioritas khususnya di Kabupaten.",
        ]
    elif var_name == "total_penduduk":
        policy_points = [
            "Kesenjangan kepadatan penduduk mempengaruhi kebutuhan dan tekanan terhadap layanan koperasi.",
            "Wilayah berpenduduk besar perlu kapasitas koperasi yang cukup untuk menyerap aktivitas ekonomi lokal.",
            "Kebijakan bisa memprioritaskan pengembangan koperasi di daerah padat penduduk dan penguatan pasar lokal di daerah dengan penduduk lebih sedikit.",
        ]
    elif var_name == "jumlah_karyawan":
        policy_points = [
            "Effect size yang relatif kecil menandakan kesenjangan jumlah karyawan koperasi antar wilayah tidak terlalu ekstrem.",
            "Fokus kebijakan dapat bergeser dari penambahan kuantitas tenaga kerja ke peningkatan kualitas SDM koperasi.",
            "Program pelatihan manajemen, keuangan, dan layanan anggota menjadi lebih relevan dibanding sekadar ekspansi jumlah karyawan.",
        ]
    else:
        policy_points = [
            "Variabel ini menunjukkan pola kesenjangan yang bisa menjadi dasar penentuan prioritas kebijakan.",
            "Periode dengan perubahan |r| paling tajam layak dijadikan bahan evaluasi program dan intervensi.",
        ]

    return html.Div(
        [
            html.P(
                [
                    html.B(var_label),
                    " dipantau dari ",
                    html.B(periode_awal),
                    " hingga ",
                    html.B(periode_akhir),
                    ".",
                ]
            ),
            html.Ul(
                [
                    html.Li(
                        f"Awal periode: |r| ≈ {r0:.3f} ({k0}) pada {periode_awal} "
                        f"– status uji {first['signif'].lower()}."
                    ),
                    html.Li(
                        f"Akhir periode: |r| ≈ {rT:.3f} ({kT}) pada {periode_akhir} "
                        f"– status uji {last['signif'].lower()}."
                    ),
                    html.Li(
                        f"Rentang keseluruhan: min |r| = {r_min:.3f}, "
                        f"max |r| = {r_max:.3f}, rata-rata ≈ {r_mean:.3f}."
                    ),
                    html.Li(
                        f"Periode dengan perbedaan signifikan: {n_signif} dari {n_total} periode (p < 0.05)."
                    ),
                    html.Li(f"Arah tren keseluruhan: {arah_tren}."),
                ]
            ),
            html.H6("Implikasi Kebijakan:", className="mt-2"),
            html.Ul([html.Li(p) for p in policy_points]),
        ]
    )


# =========================================================
# HELPER VISUALS
# =========================================================

def create_shapiro_table(df):
    """
    Buat tabel uji normalitas Shapiro–Wilk untuk semua kolom numerik utama.
    """
    rows = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if len(data) < 3:
            continue

        stat, p_val = stats.shapiro(data)
        kesimpulan = "Non-normal (p < 0.05)" if p_val < 0.05 else "Normal (p ≥ 0.05)"
        rows.append(
            {
                "Variabel": col,
                "p-value": f"{p_val:.3e}",
                "Kesimpulan": kesimpulan,
            }
        )

    if not rows:
        return html.Div("Data tidak cukup untuk uji Shapiro–Wilk.")

    return dash_table.DataTable(
        data=rows,
        columns=[
            {"name": "Variabel", "id": "Variabel"},
            {"name": "p-value", "id": "p-value"},
            {"name": "Kesimpulan", "id": "Kesimpulan"},
        ],
        style_cell={"textAlign": "left", "padding": "4px", "fontSize": 12},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        style_table={"overflowX": "auto"},
    )


def create_correlation_heatmap_strict(df):
    """Heatmap korelasi Spearman dengan anotasi angka di setiap sel."""
    corr_cols = [
        "jumlah_koperasi_aktif",
        "jumlah_koperasi_tidak_aktif",
        "jumlah_koperasi_total",
        "jumlah_karyawan",
        "jumlah_manager",
        "usaha_besar",
        "usaha_kecil",
        "usaha_menengah",
        "usaha_mikro",
        "total_penduduk",
    ]

    # pakai hanya kolom yang benar-benar ada
    available = [c for c in corr_cols if c in df.columns]

    if len(available) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Data tidak cukup untuk korelasi",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=500)
        return fig, "Data tidak cukup untuk interpretasi korelasi"

    corr_df = df[available].copy()

    # paksa numeric & drop NA
    for c in available:
        corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
    corr_df = corr_df.dropna(subset=available)

    if corr_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Data numerik valid tidak tersedia untuk korelasi",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=500)
        return fig, "Data tidak cukup untuk interpretasi korelasi"

    corr_matrix = corr_df.corr(method="spearman")

    z = corr_matrix.values.tolist()
    text_matrix = [[f"{val:.2f}" for val in row] for row in corr_matrix.values]

    x_labels = list(corr_matrix.columns)
    y_labels = list(corr_matrix.index)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale="RdBu_r",
                zmid=0,
                hoverongaps=False,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>ρ = %{z:.3f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Heatmap Korelasi Spearman - Variabel Ekonomi Koperasi",
        height=500,
        xaxis_tickangle=-45,
    )

    strongest_text = "Tidak ada korelasi kuat yang signifikan"
    max_abs = 0
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if pd.notna(val) and abs(val) > max_abs:
                max_abs = abs(val)
                strongest_text = (
                    f"🔗 Korelasi terkuat: "
                    f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]} "
                    f"(ρ={val:.3f})"
                )

    return fig, strongest_text


def create_correlation_scatter_strict(df, x_var, y_var):
    """Scatterplot korelasi Spearman dengan konversi keras ke float."""
    x_var = x_var or "jumlah_koperasi_aktif"
    y_var = y_var or "total_penduduk"

    missing = [c for c in [x_var, y_var] if c not in df.columns]
    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Kolom tidak ditemukan di dataset: {', '.join(missing)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=450)
        return fig

    cols = ["kabupaten_kota", "jenis_wilayah", x_var, y_var]
    cols = [c for c in cols if c in df.columns]
    plot_df = df[cols].copy()

    plot_df[x_var] = pd.to_numeric(plot_df[x_var], errors="coerce")
    plot_df[y_var] = pd.to_numeric(plot_df[y_var], errors="coerce")

    plot_df = plot_df.dropna(subset=[x_var, y_var])

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Data untuk kombinasi variabel ini kosong / tidak valid.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=450)
        return fig

    x_vals = plot_df[x_var].astype(float).tolist()
    y_vals = plot_df[y_var].astype(float).tolist()
    text_vals = (
        plot_df["kabupaten_kota"].tolist()
        if "kabupaten_kota" in plot_df.columns
        else None
    )

    rho = plot_df[[x_var, y_var]].corr(method="spearman").iloc[0, 1]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=8),
                text=text_vals,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{x_var}: %{{x:.0f}}<br>{y_var}: %{{y:.0f}}<extra></extra>"
                )
                if text_vals is not None
                else None,
            )
        ]
    )
    fig.update_layout(
        title=f"Scatterplot {x_var} vs {y_var} (ρ Spearman ≈ {rho:.3f})",
        height=450,
        xaxis_title=x_var,
        yaxis_title=y_var,
    )

    return fig


def create_distribution_boxplot(df, variable):
    if "jenis_wilayah" not in df.columns or variable not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Data tidak tersedia untuk boxplot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=400)
        return fig

    fig = px.box(
        df,
        x="jenis_wilayah",
        y=variable,
        color="jenis_wilayah",
        title=f"Distribusi {variable} - Perbandingan Kota vs Kabupaten",
        color_discrete_map={"Kota": "#1f77b4", "Kabupaten": "#ff7f0e"},
    )

    fig.update_layout(showlegend=False, height=400)
    return fig


def generate_statistical_results(df, method, variable):
    if method == "spearman":
        corr_matrix, p_values = calculate_spearman_correlations(df)
        if corr_matrix.empty:
            return html.Div("Data tidak cukup untuk korelasi Spearman.")

        pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                rho = corr_matrix.iloc[i, j]
                p_val = p_values.iloc[i, j] if not p_values.empty else None
                pairs.append((cols[i], cols[j], rho, p_val))

        pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]

        items = []
        for v1, v2, rho, p_val in pairs:
            txt = f"{v1} vs {v2}: ρ = {rho:.3f}"
            if p_val is not None:
                txt += f", p = {p_val:.3e}"
            items.append(html.Li(txt))

        return html.Div(
            [
                html.H5(
                    "📈 Korelasi Spearman Terkuat:", className="text-primary"
                ),
                html.Ul(items, className="mt-2"),
            ]
        )

    elif method == "mannwhitney":
        test_result = perform_mannwhitney_test(df, variable)
        if test_result:
            return html.Div(
                [
                    html.H5(
                        "🏙️ Uji Mann-Whitney U (Kota vs Kabupaten):",
                        className="text-primary",
                    ),
                    html.Table(
                        [
                            html.Tr(
                                [
                                    html.Td("Statistic U:"),
                                    html.Td(f"{test_result['statistic']:.1f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("p-value:"),
                                    html.Td(f"{test_result['p_value']:.4f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Effect Size (CLES):"),
                                    html.Td(f"{test_result['effect_size']:.3f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Interpretasi:"),
                                    html.Td(test_result["interpretation"]),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Median Kota:"),
                                    html.Td(f"{test_result['kota_median']:.1f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Median Kabupaten:"),
                                    html.Td(f"{test_result['kabupaten_median']:.1f}"),
                                ]
                            ),
                        ],
                        className="table table-sm table-striped mt-2",
                    ),
                ]
            )
        else:
            return html.Div("Data tidak cukup untuk Mann-Whitney test.")

    else:
        return html.Div("Pilih metode statistik untuk melihat hasil...")


def generate_top_regions_table(df, variable):
    top_regions = get_top_regions(df, variable, 5)

    if top_regions.empty:
        return html.Div("Data tidak tersedia")

    return dash_table.DataTable(
        data=top_regions.to_dict("records"),
        columns=[{"name": i, "id": i} for i in top_regions.columns],
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": 0}, "backgroundColor": "#d4edda"},
            {"if": {"row_index": 1}, "backgroundColor": "#f8d7da"},
            {"if": {"row_index": 2}, "backgroundColor": "#fff3cd"},
        ],
    )


def create_regional_analysis(df, variable):
    top_regions = get_top_regions(df, variable, 10)

    if top_regions.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Data tidak tersedia untuk analisis regional",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=400)
        return fig

    fig = px.bar(
        top_regions,
        x="kabupaten_kota",
        y=variable,
        title=f"Top 10 Wilayah - {variable}",
        color=variable,
        color_continuous_scale="viridis",
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
    return fig


def create_insights_cards(df, selected_periode):
    """
    Insight 1  : pakai get_statistical_insights (misal korelasi terkuat, dll.)
    Insight 2  : kenaikan & penurunan terbesar jumlah_koperasi_aktif
                 (periode sekarang vs periode sebelumnya).
    Insight 3  : pakai insight lain dari get_statistical_insights (kalau ada).
    """
    base_insights = get_statistical_insights(df) or []

    # fallback kalau utils belum ngasih apa-apa
    while len(base_insights) < 3:
        base_insights.append("📊 Data sedang dianalisis...")

    cards = []

    # ========= Insight 1 (tetap) =========
    cards.append(
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Insight 1", className="card-title"),
                        html.P(base_insights[0], className="card-text"),
                    ],
                    className="card-body",
                )
            ],
            className="card text-white bg-primary mb-3",
        )
    )

    # ========= Insight 2 (perubahan ekstrem) =========
    cards.append(create_change_extreme_insight_card(selected_periode))

    # ========= Insight 3 (tetap dari utils) =========
    cards.append(
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Insight 3", className="card-title"),
                        html.P(base_insights[2], className="card-text"),
                    ],
                    className="card-body",
                )
            ],
            className="card text-white bg-warning mb-3",
        )
    )

    return cards

def create_change_extreme_insight_card(selected_periode: str | None):
    """
    Insight 2: cari kenaikan & penurunan terbesar jumlah_koperasi_aktif
    antar-periode (periode sekarang vs periode sebelumnya) di seluruh
    kabupaten/kota.
    """
    title = "Insight 2"

    # Safety kalau periode gak valid / gak ada periode sebelumnya
    if (
        selected_periode is None
        or "periode_update" not in _df_processed_global.columns
    ):
        body = html.P(
            "Perubahan antar-periode tidak dapat dihitung karena periode tidak valid.",
            className="card-text",
        )
    else:
        df_all = _df_processed_global.copy()
        periods = sorted(df_all["periode_update"].unique().tolist())

        if selected_periode not in periods:
            body = html.P(
                "Periode terpilih tidak ditemukan dalam data.",
                className="card-text",
            )
        else:
            idx = periods.index(selected_periode)
            if idx == 0:
                body = html.P(
                    "Belum ada periode sebelumnya untuk dibandingkan "
                    f"(periode pertama: {selected_periode}).",
                    className="card-text",
                )
            else:
                prev_periode = periods[idx - 1]

                cur = df_all[df_all["periode_update"] == selected_periode][
                    ["kabupaten_kota", "jumlah_koperasi_aktif"]
                ]
                prev = df_all[df_all["periode_update"] == prev_periode][
                    ["kabupaten_kota", "jumlah_koperasi_aktif"]
                ]

                merged = cur.merge(
                    prev,
                    on="kabupaten_kota",
                    how="inner",
                    suffixes=("_cur", "_prev"),
                )

                merged["delta"] = (
                    merged["jumlah_koperasi_aktif_cur"]
                    - merged["jumlah_koperasi_aktif_prev"]
                )

                if merged.empty or merged["delta"].abs().sum() == 0:
                    body = html.P(
                        f"Perubahan jumlah koperasi aktif antara {prev_periode} "
                        f"dan {selected_periode} relatif kecil / stabil.",
                        className="card-text",
                    )
                else:
                    inc_row = merged.loc[merged["delta"].idxmax()]
                    dec_row = merged.loc[merged["delta"].idxmin()]

                    inc_delta = int(inc_row["delta"])
                    dec_delta = int(dec_row["delta"])

                    inc_text = (
                        f"Kenaikan terbesar: {inc_row['kabupaten_kota']} "
                        f"(+{inc_delta:,} koperasi aktif)."
                    )

                    if dec_delta < 0:
                        dec_text = (
                            f"Penurunan terbesar: {dec_row['kabupaten_kota']} "
                            f"({dec_delta:,} koperasi aktif)."
                        )
                    else:
                        dec_text = (
                            "Tidak ada wilayah dengan penurunan jumlah "
                            "koperasi aktif pada periode ini."
                        )

                    body = html.Div(
                        [
                            html.P(
                                f"Perbandingan jumlah_koperasi_aktif antara "
                                f"{prev_periode} dan {selected_periode}:",
                                className="small mb-1",
                            ),
                            html.Ul(
                                [
                                    html.Li(inc_text),
                                    html.Li(dec_text),
                                ],
                                className="mb-0",
                            ),
                        ]
                    )

    return html.Div(
        [
            html.Div(
                [
                    html.H6(title, className="card-title"),
                    body,
                ],
                className="card-body",
            )
        ],
        className="card text-white bg-success mb-3",
    )

def create_dataset_info(df):
    kota_count = (
        len(df[df["jenis_wilayah"] == "Kota"])
        if "jenis_wilayah" in df.columns
        else 0
    )
    kab_count = (
        len(df[df["jenis_wilayah"] == "Kabupaten"])
        if "jenis_wilayah" in df.columns
        else 0
    )

    example_period = (
        df["periode_update"].iloc[0]
        if "periode_update" in df.columns and len(df) > 0
        else "N/A"
    )

    return html.Div(
        [
            html.P(f"📁 Jumlah Observasi (baris): {len(df)}"),
            html.P(f"🏙️ Jumlah Kota: {kota_count}"),
            html.P(f"🏞️ Jumlah Kabupaten: {kab_count}"),
            html.P(f"📅 Contoh Periode Data: {example_period}"),
            html.Hr(),
            html.P("💡 Data sudah melalui preprocessing:", className="small"),
            html.Ul(
                [
                    html.Li("Standarisasi nama kolom", className="small"),
                    html.Li("Pembuatan total penduduk", className="small"),
                    html.Li(
                        "Klasifikasi jenis wilayah (Kota/Kabupaten)",
                        className="small",
                    ),
                    html.Li("Konversi variabel numerik", className="small"),
                ],
                className="small",
            ),
        ]
    )
def create_policy_summary(df, selected_periode, region_type, selected_variable):
    """
    Narasi kesimpulan & analisis kebijakan untuk periode + filter saat ini.
    Menggabungkan: normalitas, korelasi Spearman, Mann–Whitney, dan top wilayah.
    """
    if df.empty:
        return html.Div(
            "Data kosong untuk kombinasi periode dan filter wilayah yang dipilih.",
            className="text-muted",
        )

    # --- 1. Info konteks dasar ---
    periode_text = (
        f"periode {selected_periode}"
        if selected_periode is not None
        else "seluruh periode yang dianalisis"
    )

    if region_type == "Kota":
        region_label = "kota di Jawa Timur"
    elif region_type == "Kabupaten":
        region_label = "kabupaten di Jawa Timur"
    else:
        region_label = "seluruh kabupaten/kota di Jawa Timur"

    n_obs = len(df)

    # --- 2. Ringkasan normalitas (Shapiro–Wilk) ---
    normal_count = 0
    nonnormal_count = 0
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 3:
            continue
        stat, p_val = stats.shapiro(vals)
        if p_val < 0.05:
            nonnormal_count += 1
        else:
            normal_count += 1

    # --- 3. Korelasi Spearman terkuat ---
    try:
        corr_matrix, p_values = calculate_spearman_correlations(df)
    except Exception:
        corr_matrix, p_values = pd.DataFrame(), pd.DataFrame()

    strongest_pair = None
    if not corr_matrix.empty:
        cols = corr_matrix.columns.tolist()
        max_abs = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if pd.notna(val) and abs(val) > max_abs:
                    max_abs = abs(val)
                    strongest_pair = (cols[i], cols[j], val)
                    max_abs = abs(val)

    # --- 4. Hasil Mann–Whitney (kalau memungkinkan) ---
    mw_result = None
    if region_type == "all":
        try:
            mw_result = perform_mannwhitney_test(df, selected_variable)
        except Exception:
            mw_result = None

    # --- 5. Top wilayah untuk variabel utama ---
    try:
        top_regions = get_top_regions(df, selected_variable, 5)
    except Exception:
        top_regions = pd.DataFrame()

    top_text = None
    if not top_regions.empty and "kabupaten_kota" in top_regions.columns:
        first_row = top_regions.iloc[0]
        last_row = top_regions.iloc[-1]
        top_text = (
            f"Nilai tertinggi {selected_variable} terdapat di "
            f"{first_row['kabupaten_kota']}."
        )
        if len(top_regions) > 1:
            top_text += (
                f" Sementara nilai terendah di antara 5 besar berada di "
                f"{last_row['kabupaten_kota']}."
            )

    # ------------------ SUSUN NARASI ------------------ #
    bullets_rangkuman = []

    bullets_rangkuman.append(
        f"Analisis ini didasarkan pada {n_obs} observasi untuk {region_label} "
        f"pada {periode_text}."
    )

    if normal_count + nonnormal_count > 0:
        bullets_rangkuman.append(
            f"Sebagian besar variabel numerik menunjukkan distribusi "
            f"{'non-normal' if nonnormal_count >= normal_count else 'campuran normal dan non-normal'} "
            f"(≈ {nonnormal_count} variabel non-normal, {normal_count} variabel mendekati normal), "
            f"sehingga pendekatan nonparametrik seperti Spearman dan Mann–Whitney tepat digunakan."
        )

    if strongest_pair is not None:
        v1, v2, rho = strongest_pair
        abs_rho = abs(rho)
        if abs_rho < 0.10:
            strength = "sangat lemah"
        elif abs_rho < 0.30:
            strength = "lemah"
        elif abs_rho < 0.50:
            strength = "sedang"
        elif abs_rho < 0.70:
            strength = "kuat"
        else:
            strength = "sangat kuat"

        arah = "positif" if rho > 0 else "negatif"
        bullets_rangkuman.append(
            f"Korelasi rank terkuat muncul antara {v1} dan {v2} "
            f"dengan ρ ≈ {rho:.3f} ({strength}, {arah}). "
            f"Ini menggambarkan pola keterkaitan, bukan hubungan sebab-akibat langsung."
        )

    if mw_result is not None:
        bullets_rangkuman.append(
            f"Untuk variabel utama '{selected_variable}', uji Mann–Whitney "
            f"antara kota dan kabupaten menghasilkan p ≈ {mw_result['p_value']:.4f} "
            f"dengan ukuran efek (CLES) ≈ {mw_result['effect_size']:.3f}. "
            f"Median kota ≈ {mw_result['kota_median']:.1f}, sedangkan median kabupaten "
            f"≈ {mw_result['kabupaten_median']:.1f}."
        )

    if top_text is not None:
        bullets_rangkuman.append(top_text)

    # --- Implikasi kebijakan (high-level) ---
    policy_points = []

    # poin generik berbasis korelasi
    if strongest_pair is not None:
        v1, v2, rho = strongest_pair
        if abs(rho) >= 0.3:
            policy_points.append(
                f"Pasangan variabel {v1}–{v2} dapat dijadikan fokus monitoring bersama: "
                f"perubahan pada salah satu indikator berpotensi diikuti perubahan pola pada indikator lain."
            )

    # poin berbasis Mann–Whitney
    if mw_result is not None and mw_result["p_value"] < 0.05:
        if mw_result["kabupaten_median"] > mw_result["kota_median"]:
            arah_kesenjangan = "kabupaten cenderung berada di atas kota"
        else:
            arah_kesenjangan = "kota cenderung berada di atas kabupaten"

        policy_points.append(
            f"Kesenjangan {selected_variable} antara kota dan kabupaten signifikan; "
            f"{arah_kesenjangan} untuk indikator ini. Program penguatan bisa diprioritaskan "
            f"di kelompok wilayah yang tertinggal (median lebih rendah)."
        )

    # poin berbasis konsentrasi wilayah
    if top_text is not None:
        policy_points.append(
            "Wilayah dengan nilai tertinggi dapat dijadikan rujukan praktik baik, "
            "sementara wilayah di bawah rata-rata perlu pendampingan lebih intensif."
        )

    if not policy_points:
        policy_points.append(
            "Temuan saat ini lebih bersifat deskriptif; perlu analisis lanjutan "
            "dan triangulasi dengan informasi kualitatif sebelum dijadikan dasar kebijakan spesifik."
        )

    return html.Div(
        [
            html.P("Ringkasan utama:", className="fw-bold mb-1"),
            html.Ul(
                [html.Li(p, className="mb-1") for p in bullets_rangkuman],
                className="mb-2",
            ),
            html.P("Implikasi kebijakan (awal):", className="fw-bold mb-1"),
            html.Ul(
                [html.Li(p, className="mb-1") for p in policy_points],
                className="mb-0",
            ),
            html.P(
                "Catatan: interpretasi ini berbasis korelasi dan perbandingan distribusi, "
                "bukan bukti kausal.",
                className="mt-2 text-muted",
                style={"fontSize": "0.8rem"},
            ),
        ]
    )
    
def create_period_highlights(df, selected_periode, region_type):
    """
    Highlight antar-periode sederhana:
    - Variabel dengan kenaikan persentase tertinggi
    - Variabel dengan penurunan persentase terbesar
    """
    if selected_periode is None or "periode_update" not in _df_processed_global.columns:
        return html.Div(
            "Highlight antar-periode belum tersedia untuk kombinasi ini.",
            className="text-muted",
        )

    if selected_periode not in PERIODE_LIST:
        return html.Div(
            "Periode tidak dikenali dalam data global.", className="text-muted"
        )

    idx = PERIODE_LIST.index(selected_periode)
    if idx == 0:
        return html.Div(
            "Tidak ada periode sebelumnya untuk dibandingkan (ini periode paling awal).",
            className="text-muted",
        )

    prev_periode = PERIODE_LIST[idx - 1]

    # Filter global untuk current & prev
    base = _df_processed_global.copy()
    if region_type != "all" and "jenis_wilayah" in base.columns:
        base = base[base["jenis_wilayah"] == region_type]

    cur_df = base[base["periode_update"] == selected_periode]
    prev_df = base[base["periode_update"] == prev_periode]

    if cur_df.empty or prev_df.empty:
        return html.Div(
            "Data untuk periode saat ini atau sebelumnya tidak lengkap untuk highlight.",
            className="text-muted",
        )

    deltas = []
    for col in NUMERIC_COLS:
        if col not in cur_df.columns or col not in prev_df.columns:
            continue

        cur_mean = cur_df[col].astype(float).mean()
        prev_mean = prev_df[col].astype(float).mean()

        if not np.isfinite(cur_mean) or not np.isfinite(prev_mean) or prev_mean == 0:
            continue

        delta_pct = (cur_mean - prev_mean) / prev_mean * 100.0
        deltas.append(
            {
                "var": col,
                "delta_pct": delta_pct,
                "cur_mean": cur_mean,
                "prev_mean": prev_mean,
            }
        )

    if not deltas:
        return html.Div(
            "Tidak ditemukan perubahan yang dapat dihitung antar-periode.",
            className="text-muted",
        )

    # cari naik & turun terbesar
    inc = max(deltas, key=lambda d: d["delta_pct"])
    dec = min(deltas, key=lambda d: d["delta_pct"])

    def fmt_var(v):
        return VAR_NAME_MAP.get(v, v.replace("_", " ").title())

    periode_label = f"{prev_periode} → {selected_periode}"

    items = [
        html.Li(
            f"Perbandingan dilakukan antara {periode_label} "
            f"untuk {'semua wilayah' if region_type == 'all' else region_type}.",
            className="mb-1",
        ),
        html.Li(
            f"Kenaikan relatif terbesar: {fmt_var(inc['var'])} "
            f"(≈ {inc['delta_pct']:+.1f}% dibanding periode sebelumnya).",
            className="mb-1",
        ),
        html.Li(
            f"Penurunan relatif terbesar: {fmt_var(dec['var'])} "
            f"(≈ {dec['delta_pct']:+.1f}% dibanding periode sebelumnya).",
            className="mb-1",
        ),
    ]

    return html.Div(
        [
            html.P("Highlight antar-periode:", className="fw-bold mb-1"),
            html.Ul(items, className="mb-0"),
        ]
    )

def create_mini_trend_main_var(global_df, selected_variable, region_type):
    """
    Sparkline sederhana: rata-rata variabel utama per periode (maks 6 periode terakhir).
    """
    var = selected_variable or "jumlah_koperasi_aktif"
    df = global_df.copy()

    if var not in df.columns or "periode_update" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Data tren tidak tersedia.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=180, margin=dict(l=30, r=10, t=30, b=30))
        return fig

    if region_type != "all" and "jenis_wilayah" in df.columns:
        df = df[df["jenis_wilayah"] == region_type]

    tmp = (
        df.groupby("periode_update")[var]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={var: "mean_val"})
    )

    if tmp.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Data tren tidak tersedia.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=180, margin=dict(l=30, r=10, t=30, b=30))
        return fig

    # ambil 6 periode terakhir
    tmp = tmp.tail(6)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tmp["periode_update"],
            y=tmp["mean_val"],
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate="Periode %{x}<br>Rata-rata ≈ %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Rata-rata {VAR_NAME_MAP.get(var, var)} (≤ 6 periode terakhir)",
        height=190,
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title="Periode",
        yaxis_title="Rata-rata",
        xaxis_tickangle=-30,
    )

    return fig

def create_mini_jenis_usaha_figure(df):
    """
    Komposisi usaha besar/kecil/menengah/mikro di periode & filter saat ini.
    """
    cols_usaha = ["usaha_besar", "usaha_kecil", "usaha_menengah", "usaha_mikro"]
    available = [c for c in cols_usaha if c in df.columns]

    if not available:
        fig = go.Figure()
        fig.add_annotation(
            text="Data jenis usaha tidak tersedia.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=220, margin=dict(l=30, r=10, t=30, b=30))
        return fig

    agg = df[available].astype(float).sum().reset_index()
    agg.columns = ["jenis_usaha", "jumlah"]

    total = agg["jumlah"].sum()
    if total <= 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Nilai jenis usaha nol/invalid.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=220, margin=dict(l=30, r=10, t=30, b=30))
        return fig

    agg["persen"] = agg["jumlah"] / total * 100.0

    fig = px.bar(
        agg,
        x="jenis_usaha",
        y="jumlah",
        text=agg["persen"].map(lambda v: f"{v:.1f}%"),
        title="Distribusi Jenis Usaha (total = 100%)",
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>Jumlah: %{y:.0f}<br>Share: %{text}<extra></extra>",
    )
    fig.update_layout(
        height=220,
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title="Jenis usaha",
        yaxis_title="Jumlah",
    )

    return fig

def apply_dark_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        xaxis=dict(
            gridcolor="#1f2937",
            zerolinecolor="#1f2937",
            tickfont=dict(color="#e5e7eb"),
            titlefont=dict(color="#e5e7eb"),
        ),
        yaxis=dict(
            gridcolor="#1f2937",
            zerolinecolor="#1f2937",
            tickfont=dict(color="#e5e7eb"),
            titlefont=dict(color="#e5e7eb"),
        ),
        legend=dict(
            font=dict(color="#e5e7eb"),
        ),
    )
    return fig

