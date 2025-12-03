# components/time_series.py

from dash import dcc, html, Input, Output, callback  # type: ignore
import dash_bootstrap_components as dbc # type: ignore
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from utils.data_loader import load_tsa_data, calculate_tsa_metrics


def create_tsa_tab():
    """TSA Tab untuk analisis penumpang bus"""

    # Load data
    hist_df, forecast_df = load_tsa_data()
    metrics = calculate_tsa_metrics(hist_df, forecast_df)

    # =========================================================
    # >>> 1. Agregasi data TAHUNAN untuk KPI growth dinamis
    # =========================================================
    hist_df = hist_df.copy()
    hist_df["year"] = hist_df["periode"].dt.year

    yearly_df = (
        hist_df.groupby("year")["actual"]
        .sum()
        .reset_index()  # kolom: ['year', 'actual']
    )
    # simpan dalam bentuk list of dict agar bisa masuk ke dcc.Store (JSON serializable)
    yearly_data = yearly_df.to_dict("records")

    return html.Div(
        [
            html.H3(
                "📈 Analisis Time Series - Penumpang Bus Terminal Tipe B Jawa Timur",
                className="mt-4",
            ),

            # 1. KPI CARDS (sekarang kirim yearly_data juga)
            create_kpi_cards(metrics, yearly_data),

            # >>> Store data tahunan untuk dipakai callback growth
            dcc.Store(id="tsa-yearly-data", data=yearly_data),

            # 2. MAIN VISUALIZATION - TREND & FORECAST
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.H4(
                            "Trend Historis & Forecast Penumpang Bus",
                            className="mb-0",
                        )
                    ),
                    dbc.CardBody(
                        [
                            dcc.Graph(
                                id="bus-trend-forecast-plot",
                                figure=create_trend_forecast_plot(
                                    hist_df, forecast_df
                                ),
                            )
                        ]
                    ),
                ],
                className="mt-4",
            ),

            # 3. INTERPRETATION
            dbc.Alert(
                [
                    html.H5(
                        "📊 Interpretasi Trend Penumpang Bus:",
                        className="tsa-title",
                    ),
                    html.Ul(
                        [
                            html.Li("🚀 Puncak 2.5 juta penumpang (Des 2019)"),
                            html.Li("💥 Dampak COVID: Turun 98% (Mei 2020)"),
                            html.Li("📈 Pemulihan bertahap 2021–2023"),
                            html.Li("🎯 Forecast 2025: Stabil di ~1.1–1.5 juta/bulan"),
                        ],
                        className="tsa-list",
                    ),
                ],
                color="dark",              # jangan 'light' lagi
                className="mt-3 tsa-box",  # tambahin class custom
            ),

        ]
    )


def create_kpi_cards(metrics, yearly_data):
    """Buat KPI cards untuk summary statistics + GROWTH dinamis"""

    # ============================
    # Daftar tahun & default: min → max (misal 2019 → 2025)
    # ============================
    years = sorted({int(row["year"]) for row in yearly_data}) if yearly_data else []
    if years:
        default_from = years[0]       # tahun paling awal (contoh: 2019)
        default_to = years[-1]        # tahun paling akhir (contoh: 2025)
    else:
        default_from = default_to = datetime.now().year

    return dbc.Row(
        [
            # KPI 1: Rata-rata
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    f"{metrics.get('avg_passengers', 0):,.0f}",
                                    className="card-title text-primary",
                                ),
                                html.P(
                                    "Rata-rata Penumpang/Bulan",
                                    className="card-text",
                                ),
                            ]
                        )
                    ],
                    color="light",
                ),
                width=3,
            ),

            # KPI 2: Puncak
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    f"{metrics.get('peak_passengers', 0):,.0f}",
                                    className="card-title text-success",
                                ),
                                html.P(
                                    f"Puncak Tertinggi ({metrics.get('peak_month', 'N/A')})",
                                    className="card-text",
                                ),
                            ]
                        )
                    ],
                    color="light",
                ),
                width=3,
            ),

            # KPI 3: Terendah
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H4(
                                    f"{metrics.get('lowest_passengers', 0):,.0f}",
                                    className="card-title text-warning",
                                ),
                                html.P(
                                    f"Terendah ({metrics.get('lowest_month', 'N/A')})",
                                    className="card-text",
                                ),
                            ]
                        )
                    ],
                    color="light",
                ),
                width=3,
            ),

            # KPI 4: Growth dinamis (tanpa subtitle tahun spesifik)
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H6(
                                    "Growth Penumpang", className="text-muted"
                                ),
                                # Nilai growth yang akan di-update callback
                                html.H4(
                                    id="tsa-growth-kpi-value",
                                    children=f"{metrics.get('growth_rate', 0):+.1f}%",
                                    className="card-title text-info",
                                ),
                                # Subtitle statis (bukan "Growth 2025 vs 2024" lagi)
                                html.P(
                                    "Berdasarkan tahun yang dipilih",
                                    className="card-text",
                                ),
                                html.Div(className="mt-2"),
                                html.Div(
                                    [
                                        html.Span("Dari:", className="me-1"),
                                        dcc.Dropdown(
                                            id="tsa-growth-year-from",
                                            options=[
                                                {"label": str(y), "value": int(y)}
                                                for y in years
                                            ],
                                            value=int(default_from),
                                            clearable=False,
                                            style={
                                                "width": "110px",
                                                "display": "inline-block",
                                            },
                                        ),
                                    ],
                                    className="mb-1",
                                ),
                                html.Div(
                                    [
                                        html.Span("Ke:", className="me-1"),
                                        dcc.Dropdown(
                                            id="tsa-growth-year-to",
                                            options=[
                                                {"label": str(y), "value": int(y)}
                                                for y in years
                                            ],
                                            value=int(default_to),
                                            clearable=False,
                                            style={
                                                "width": "110px",
                                                "display": "inline-block",
                                            },
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    color="light",
                ),
                width=3,
            ),
        ],
        className="mb-4",
    )

def create_trend_forecast_plot(hist_df, forecast_df):
    """Buat plot trend historis + forecast dengan confidence intervals - FIXED VERSION"""

    # KONVERSI KE LIST
    x_hist = hist_df["periode"].dt.to_pydatetime().tolist()
    y_hist = hist_df["actual"].tolist()

    x_fore = forecast_df["Bulan"].dt.to_pydatetime().tolist()
    y_fore = forecast_df["forecast"].tolist()
    upper_95 = forecast_df["upper_95"].tolist()
    lower_95 = forecast_df["lower_95"].tolist()
    upper_80 = forecast_df["upper_80"].tolist()
    lower_80 = forecast_df["lower_80"].tolist()

    # Sambung garis historis → forecast
    last_hist_date = x_hist[-1]
    last_hist_val = y_hist[-1]
    x_fore_line = [last_hist_date] + x_fore
    y_fore_line = [last_hist_val] + y_fore

    fig = go.Figure()

    # 1. Historical Data (Garis biru)
    fig.add_trace(
        go.Scatter(
            x=x_hist,
            y=y_hist,
            mode="lines+markers",
            name="Data Aktual",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=4),
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} penumpang<extra></extra>",
        )
    )

    # 2. Confidence Interval 95%
    fig.add_trace(
        go.Scatter(
            x=x_fore + x_fore[::-1],
            y=upper_95 + lower_95[::-1],
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.15)",
            line=dict(width=0),
            name="95% Confidence",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # 3. Confidence Interval 80%
    fig.add_trace(
        go.Scatter(
            x=x_fore + x_fore[::-1],
            y=upper_80 + lower_80[::-1],
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.3)",
            line=dict(width=0),
            name="80% Confidence",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # 4. Forecast Data (Garis orange putus-putus)
    fig.add_trace(
        go.Scatter(
            x=x_fore_line,
            y=y_fore_line,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            marker=dict(size=4, symbol="diamond"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f} penumpang<extra></extra>",
        )
    )

    # 5. COVID Impact Annotation
    fig.add_annotation(
        x=pd.Timestamp("2020-05-01"),
        y=37444,
        text="Dampak COVID-19<br>Turun 98%",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-50,
        bgcolor="red",
        opacity=0.8,
    )

    # Dynamic Y-axis range
    all_vals = y_hist + upper_95 + lower_95 + upper_80 + lower_80
    y_min = min(all_vals) * 0.8
    y_max = max(all_vals) * 1.1

    # Layout settings
    fig.update_layout(
        title="Trend Penumpang Bus Terminal Tipe B Jawa Timur (2019-2025)",
        xaxis_title="Periode",
        yaxis_title="Jumlah Penumpang",
        hovermode="x unified",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(range=[y_min, y_max]),
    )

    # Format axes
    fig.update_yaxes(tickformat=",", gridcolor="lightgray")
    fig.update_xaxes(gridcolor="lightgray", tickformat="%b %Y", dtick="M6")

    return fig


# =========================================================
# >>> 3. CALLBACK: Hitung growth berdasarkan pilihan tahun
# =========================================================
@callback(
    Output("tsa-growth-kpi-value", "children"),
    Output("tsa-growth-kpi-value", "className"),
    Input("tsa-growth-year-from", "value"),
    Input("tsa-growth-year-to", "value"),
    Input("tsa-yearly-data", "data"),
)
def update_growth_kpi(year_from, year_to, yearly_data):
    # Safety check
    if not yearly_data or year_from is None or year_to is None:
        return "–", "card-title text-muted"

    # mapping {tahun: total_penumpang}
    year_map = {int(row["year"]): float(row["actual"]) for row in yearly_data}

    if year_from not in year_map or year_to not in year_map:
        return "–", "card-title text-muted"

    if year_from == year_to:
        return "0.0%", "card-title text-secondary"

    base = year_map[year_from]
    comp = year_map[year_to]

    if base <= 0:
        return "–", "card-title text-muted"

    growth = (comp - base) / base * 100.0
    value_str = f"{growth:+.1f}%"

    # warna: hijau naik, merah turun, abu-abu netral
    if growth > 0:
        cls = "card-title text-success"
    elif growth < 0:
        cls = "card-title text-danger"
    else:
        cls = "card-title text-secondary"

    return value_str, cls

