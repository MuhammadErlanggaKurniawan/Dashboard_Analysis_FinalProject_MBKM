# components/pattern_recognition.py

from dash import html, dcc, callback, Input, Output # type: ignore
import dash_bootstrap_components as dbc # type: ignore
import dash_leaflet as dl # type: ignore
import plotly.graph_objects as go
import pandas as pd
from shapely import wkt # type: ignore
from shapely.geometry import mapping # type: ignore
from dash_extensions.javascript import assign # type: ignore

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
_geo_df["Cluster"] = _geo_df["Cluster"].astype(int)
print("[DEBUG] Columns in _geo_df:", _geo_df.columns.tolist())
print("[DEBUG] Cluster value counts:\n", _geo_df["Cluster"].value_counts(dropna=False))
_radar_source = build_radar_source()
_sil_source = build_silhouette_source()


# ============================
# 2. Konversi WKT → GeoJSON
# ============================
CLUSTER_COLORS = {
    0: "#f97316",  # orange – Mikro-Intensif
    1: "#6366f1",  # indigo – Struktural-Besar
}

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
                # langsung tempel warna di properties
                "fillColor": CLUSTER_COLORS.get(cl_int, "#9ca3af"),
            },
            "geometry": geom_geojson,
        }
        features.append(feature)

    # DEBUG: cek satu contoh fitur, setelah features terisi
    if features:
        print(
            "[DEBUG] example cluster:",
            features[0]["properties"]["cluster"],
            type(features[0]["properties"]["cluster"]),
        )

    fc = {"type": "FeatureCollection", "features": features}
    print(f"[DEBUG] GeoJSON FeatureCollection built: {len(features)} features")
    return fc



_all_geojson_data = prepare_all_geojson_data()


# ============================
# 3. JavaScript style & tooltip
# ============================

# warna cluster (SATU-SATUNYA sumber kebenaran warna)
# warna cluster (SATU-SATUNYA sumber kebenaran warna)
CLUSTER_COLORS = {
    0: "#f97316",  # orange – Mikro-Intensif
    1: "#6366f1",  # indigo – Struktural-Besar
}


def _build_colors_js():
    pairs = []
    for cid, col in CLUSTER_COLORS.items():
        pairs.append(f"{cid}: '{col}'")     # numeric
        pairs.append(f"'{cid}': '{col}'")  # string
    return "{%s}" % ", ".join(pairs)


geojson_style = assign(
    """
function(feature, context){
    return {
        color: '#0f172a',
        weight: 1,
        fillColor: feature.properties.fillColor || '#9ca3af',
        fillOpacity: 0.8
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
        color: '#e5e7eb',
        fillOpacity: 0.95
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
                                            # kotak warna cluster 0
                                            html.Span(
                                                style={
                                                    "display": "inline-block",
                                                    "width": "14px",
                                                    "height": "14px",
                                                    "borderRadius": "3px",
                                                    "backgroundColor": CLUSTER_COLORS[0],
                                                    "border": "1px solid #e5e7eb",
                                                    "marginRight": "8px",
                                                }
                                            ),
                                            html.Span(
                                                CLUSTER_LABELS[0],
                                                style={"color": "#e5e7eb"},
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        [
                                            # kotak warna cluster 1
                                            html.Span(
                                                style={
                                                    "display": "inline-block",
                                                    "width": "14px",
                                                    "height": "14px",
                                                    "borderRadius": "3px",
                                                    "backgroundColor": CLUSTER_COLORS[1],
                                                    "border": "1px solid #e5e7eb",
                                                    "marginRight": "8px",
                                                }
                                            ),
                                            html.Span(
                                                CLUSTER_LABELS[1],
                                                style={"color": "#e5e7eb"},
                                            ),
                                        ],
                                        className="mb-2",
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
        color = CLUSTER_COLORS.get(int(cl), "#9ca3af")

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=CLUSTER_LABELS.get(int(cl), f"Cluster {int(cl)}"),
                line=dict(color=color, width=2),
                fillcolor=color,   # pakai hex biasa
                opacity=0.6,       # transparansi diatur di sini
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#020617",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="#1e293b",
                linecolor="#475569",
                tickfont=dict(color="#e5e7eb"),
            ),
            angularaxis=dict(
                gridcolor="#1e293b",
                tickfont=dict(color="#e5e7eb"),
            ),
        ),
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#1e293b",
            borderwidth=1,
        ),
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
    colors = [
        CLUSTER_COLORS.get(int(c), "#64748b") 
        for c in df["Cluster"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["Silhouette"],
            y=df["KABKOT"],
            orientation="h",
            marker=dict(color=colors),
        )
    )

    fig.update_layout(
        xaxis_title="Nilai Silhouette",
        yaxis_title="Kabupaten/Kota",
        margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        xaxis=dict(
            gridcolor="#1e293b",
            linecolor="#475569",
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            linecolor="#475569",
        ),
    )

    return fig
