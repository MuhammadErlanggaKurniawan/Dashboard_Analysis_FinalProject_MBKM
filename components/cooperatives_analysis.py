# components/cooperatives_analysis.py

import dash
from dash import dcc, html, Input, Output, callback, dash_table
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
                                    for v in (_mw_trend_df["variabel"].unique().tolist()
                                              if not _mw_trend_df.empty else [])
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
                            html.H5("Ringkasan Singkat", className="mb-2"),
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
            empty_fig,
            empty_fig,
            html.Div(msg),
            html.Div(msg),
            empty_fig,
            [],
            html.Div(msg),
            msg,
            empty_fig,
            html.Div(msg),
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
    insights_cards = create_insights_cards(df_processed)
    dataset_info = create_dataset_info(df_processed)

    corr_interpretation = strongest_text  # teks di bawah heatmap

    return (
        heatmap_fig,
        box_fig,
        stats_results,
        top_regions_table,
        regional_fig,
        insights_cards,
        dataset_info,
        corr_interpretation,
        scatter_fig,
        shapiro_table,
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
            line=dict(color="lightgray", dash="dot", width=1),
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

    first = df.iloc[0]
    last = df.iloc[-1]

    var_label = VAR_LABELS.get(var_name, var_name)

    r0 = first["r_abs"]
    rT = last["r_abs"]
    k0 = first["kategori_r"]
    kT = last["kategori_r"]

    signif0 = first["signif"].lower()
    signifT = last["signif"].lower()

    return [
        html.P(
            [
                html.B(var_label),
                " dipantau dari ",
                html.B(str(first["periode"])),
                " hingga ",
                html.B(str(last["periode"])),
                ".",
            ]
        ),
        html.Ul(
            [
                html.Li(
                    f"Awal periode: |r| ≈ {r0:.3f} ({k0}) pada {first['periode']} "
                    f"({signif0})."
                ),
                html.Li(
                    f"Akhir periode: |r| ≈ {rT:.3f} ({kT}) pada {last['periode']} "
                    f"({signifT})."
                ),
            ]
        ),
    ]


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


def create_insights_cards(df):
    insights = get_statistical_insights(df)

    if not insights:
        insights = [
            "📊 Data sedang dianalisis...",
            "🔍 Gunakan filter untuk melihat insights spesifik",
            "💡 Pilih variabel dan metode statistik yang berbeda",
        ]

    colors = ["primary", "success", "warning"]
    cards = []
    for i, insight in enumerate(insights[:3]):
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.H6(
                                f"Insight {i + 1}", className="card-title"
                            ),
                            html.P(insight, className="card-text"),
                        ],
                        className="card-body",
                    )
                ],
                className=f"card text-white bg-{colors[i]} mb-3",
            )
        )

    return cards


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
