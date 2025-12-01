# components/pattern_recognition.py

from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import plotly.graph_objects as go
import pandas as pd
from shapely import wkt
from shapely.geometry import mapping
from dash_extensions.javascript import assign

from utils.pattern_data import (
    load_geo_table_and_geojson,
    build_radar_source,
    build_silhouette_source,
    compute_overall_silhouette,
    CLUSTER_LABELS,
    RADAR_FEATURES,
)

# ============================
# 1. Preload data
# ============================

_geo_df = load_geo_table_and_geojson()
_radar_source = build_radar_source()
_sil_source = build_silhouette_source()


# ============================
# 2. Konversi WKT → GeoJSON
# ============================

def prepare_all_geojson_data():
    """
    Bangun FeatureCollection GeoJSON dari _geo_df.
    Properti penting:
    - kabkot
    - cluster (int)
    - cluster_label (string, pakai CLUSTER_LABELS)
    """
    features = []

    if "geometry" not in _geo_df.columns:
        print("[WARN] Kolom 'geometry' tidak ada di hasil_clustering.csv – peta tidak akan muncul.")
        return {"type": "FeatureCollection", "features": []}

    for idx, row in _geo_df.iterrows():
        geom_wkt = row.get("geometry", None)
        if pd.isna(geom_wkt):
            continue

        try:
            geom = wkt.loads(geom_wkt)
            geom_geojson = mapping(geom)
        except Exception as e:
            print(f"[WARN] Gagal parse WKT untuk index {idx}: {e}")
            continue

        cl_raw = row.get("Cluster", 0)
        try:
            cl_int = int(cl_raw)
        except Exception:
            cl_int = 0

        feature = {
            "type": "Feature",
            "id": str(idx),
            "properties": {
                "kabkot": row.get("kabkot", ""),
                "cluster": cl_int,
                "cluster_label": CLUSTER_LABELS.get(cl_int, f"Cluster {cl_int}"),
            },
            "geometry": geom_geojson,
        }
        features.append(feature)

    fc = {"type": "FeatureCollection", "features": features}
    print(f"[DEBUG] GeoJSON FeatureCollection built: {len(features)} features")
    return fc


_all_geojson_data = prepare_all_geojson_data()


# ============================
# 3. JavaScript style & tooltip
# ============================

# warna cluster
CLUSTER_COLORS = {
    0: "#1f77b4",  # biru
    1: "#ff7f0e",  # oranye
}

# fungsi JS untuk style poligon
geojson_style = assign(
    """
function(feature, context){
    const cl = feature.properties.cluster;
    const colors = {0: '#1f77b4', 1: '#ff7f0e'};
    return {
        color: '#ffffff',
        weight: 1,
        fillColor: colors[cl] || '#808080',
        fillOpacity: 0.7
    };
}
"""
)

# fungsi JS untuk hover style
hover_style = assign(
    """
function(feature, context){
    return {
        weight: 3,
        color: '#000000',
        fillOpacity: 0.9
    };
}
"""
)

# fungsi JS untuk tooltip per feature
on_each_feature = assign(
    """
function(feature, layer, context){
    const kabkot = feature.properties.kabkot;
    const label = feature.properties.cluster_label;
    layer.bindTooltip(kabkot + ' – ' + label, {sticky: true});
}
"""
)


# ============================
# 4. Layout
# ============================

def create_pattern_layout():
    # opsi dropdown cluster
    cluster_options = [{"label": "Semua Cluster", "value": "all"}]
    if "Cluster" in _geo_df.columns:
        cluster_options += [
            {
                "label": CLUSTER_LABELS.get(int(c), f"Cluster {int(c)}"),
                "value": str(int(c)),
            }
            for c in sorted(_geo_df["Cluster"].dropna().unique())
        ]

    overall_sil = compute_overall_silhouette()
    if overall_sil == overall_sil:  # bukan NaN
        sil_text = f"Rata-rata silhouette seluruh cluster: {overall_sil:.3f}"
    else:
        sil_text = "Rata-rata silhouette seluruh cluster: -"

    return html.Div(
        [
            html.H2(
                "🔍 Analisis Pengenalan Pola – Koperasi Jawa Timur",
                className="mb-2",
            ),
            html.P(
                "Halaman ini menampilkan hasil clustering koperasi per kabupaten/kota "
                "di Jawa Timur menggunakan metode Agglomerative (complete–cosine, k=2). "
                "Gunakan filter di bawah untuk mengeksplorasi profil tiap cluster.",
                className="text-muted mb-3",
            ),
            # filter cluster
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Filter Cluster"),
                            dcc.Dropdown(
                                id="pattern-cluster-filter",
                                options=cluster_options,
                                value="all",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    )
                ],
                className="mb-3",
            ),
            # Peta + ringkasan
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Peta Sebaran Cluster Koperasi"),
                            dl.Map(
                                [
                                    dl.TileLayer(),
                                    dl.GeoJSON(
                                        id="geojson-layer",
                                        data=_all_geojson_data,
                                        options=dict(
                                            style=geojson_style,
                                            onEachFeature=on_each_feature,
                                        ),
                                        hoverStyle=hover_style,
                                        zoomToBounds=True,
                                    ),
                                ],
                                id="leaflet-map",
                                center=[-7.5, 112.5],
                                zoom=7,
                                style={"height": "420px", "width": "100%"},
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.H4("Ringkasan Silhouette Score"),
                            html.P(
                                sil_text,
                                id="pattern-silhouette-summary",
                                className="lead fw-semibold",
                            ),
                            html.P(
                                "Semakin tinggi nilai silhouette (mendekati 1), "
                                "semakin jelas pemisahan antar cluster. "
                                "Nilai mendekati 0 artinya batas cluster kurang tegas.",
                                className="text-muted",
                            ),
                            html.H5("Legenda Cluster:", className="mt-3"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "■",
                                                style={
                                                    "color": CLUSTER_COLORS[0],
                                                    "fontSize": "20px",
                                                },
                                            ),
                                            html.Span(
                                                f" {CLUSTER_LABELS[0]}",
                                                className="ms-2",
                                            ),
                                        ],
                                        className="mb-1",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "■",
                                                style={
                                                    "color": CLUSTER_COLORS[1],
                                                    "fontSize": "20px",
                                                },
                                            ),
                                            html.Span(
                                                f" {CLUSTER_LABELS[1]}",
                                                className="ms-2",
                                            ),
                                        ],
                                        className="mb-1",
                                    ),
                                ]
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-4",
            ),
            # Radar + silhouette per kab/kota
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Profil Rata-Rata Tiap Cluster (Radar Chart)"),
                            dcc.Graph(
                                id="pattern-radar",
                                style={"height": "420px"},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Silhouette per Kabupaten/Kota"),
                            dcc.Graph(
                                id="pattern-silhouette",
                                style={"height": "420px"},
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
        ],
        className="p-3",
    )


# ============================
# 5. Callbacks
# ============================

@callback(
    Output("geojson-layer", "data"),
    Input("pattern-cluster-filter", "value"),
)
def update_geojson_data(cluster_value):
    """Filter fitur peta berdasarkan cluster dropdown."""
    if cluster_value in (None, "all"):
        return _all_geojson_data

    try:
        cl = int(cluster_value)
    except ValueError:
        return _all_geojson_data

    feats = [
        f
        for f in _all_geojson_data["features"]
        if f["properties"]["cluster"] == cl
    ]
    return {"type": "FeatureCollection", "features": feats}


@callback(
    Output("pattern-radar", "figure"),
    Input("pattern-cluster-filter", "value"),  # cuma buat trigger
)
def update_pattern_radar(_cluster_value):
    df = _radar_source.copy()
    if df.empty:
        return go.Figure()

    categories = RADAR_FEATURES + [RADAR_FEATURES[0]]

    fig = go.Figure()
    for cl in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cl].set_index("feature")["value"]
        values = [sub[f] for f in RADAR_FEATURES] + [sub[RADAR_FEATURES[0]]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=CLUSTER_LABELS.get(int(cl), f"Cluster {int(cl)}"),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


@callback(
    Output("pattern-silhouette", "figure"),
    Input("pattern-cluster-filter", "value"),
)
def update_pattern_silhouette(cluster_value):
    df = _sil_source.copy()

    if cluster_value not in (None, "all"):
        try:
            cl = int(cluster_value)
            df = df[df["Cluster"] == cl]
        except ValueError:
            pass

    if "Silhouette" not in df.columns or df["Silhouette"].isna().all():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Nilai Silhouette",
            yaxis_title="Kabupaten/Kota",
            margin=dict(l=10, r=10, t=20, b=20),
        )
        return fig

    df = df.sort_values("Silhouette", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["Silhouette"],
            y=df["KABKOT"],
            orientation="h",
        )
    )
    fig.update_layout(
        xaxis_title="Nilai Silhouette",
        yaxis_title="Kabupaten/Kota",
        margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig
