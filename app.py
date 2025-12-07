# app.py
import dash  # type: ignore
from dash import dcc, html, Input, Output, callback  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import dash_leaflet  # type: ignore  # (dipakai di komponen lain)

from components.navbar import create_navbar
from components.time_series import create_tsa_tab
from components.cooperatives_analysis import create_cooperatives_layout
from components.pattern_recognition import create_pattern_layout

app = dash.Dash(
    __name__,
    external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",],
    suppress_callback_exceptions=True,
)
app.title = "Economic Analysis Dashboard"
server = app.server


# =====================================
# 1. PAGE 1 – ABOUT / LANDING
# =====================================
def create_about_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    # LEFT SIDE
                    dbc.Col(
                        [
                            html.H1(
                                "👋 Hello, I'm Angga",
                                className="text-white mb-3",
                                style={"fontSize": "3rem", "fontWeight": "700"}
                            ),
                            html.H4(
                                "Data Scientist At Dinas Komunikasi dan Informatika Provinsi Jawa Timur",
                                className="text-info mb-4",
                                style={"fontSize": "1.7rem", "fontWeight": "600"}
                            ),

                            html.P(
                                """
                                Saya Muhammad Erlangga Kurniawan, mahasiswa Sains Data UPN 'Veteran'
                                Jawa Timur yang fokus pada analisis data, machine learning, dan
                                pengembangan dashboard interaktif. Project magang saya di KOMINFO
                                Provinsi Jawa Timur mencakup analisis koperasi & UMKM, time series
                                penumpang Terminal Tipe B, serta clustering struktur koperasi per wilayah.
                                """,
                                className="text-light",
                                style={"fontSize": "1.15rem"},
                            ),

                            html.P(
                                """
                                Dashboard ini merangkum tiga analisis utama: nonparametrik koperasi,
                                peramalan deret waktu, dan pengenalan pola (clustering). Klik tombol
                                di bawah untuk masuk ke dashboard analisis.
                                """,
                                className="text-light",
                                style={"fontSize": "1.15rem"},
                            ),

                            # ===== SOCIAL LINKS =====
                            html.Div(
                                [
                                    html.A(
                                        html.I(className="bi bi-github"),
                                        href="https://github.com/MuhammadErlanggaKurniawan",
                                        target="_blank",
                                        className="social-icon",
                                    ),
                                    html.A(
                                        html.I(className="bi bi-linkedin"),
                                        href="https://www.linkedin.com/in/muhammad-erlangga-kurniawan-010a43292/",
                                        target="_blank",
                                        className="social-icon",
                                    ),
                                    html.A(
                                        html.I(className="bi bi-instagram"),
                                        href="https://www.instagram.com/angg4aaa/",
                                        target="_blank",
                                        className="social-icon",
                                    ),
                                    html.A(
                                        html.I(className="bi bi-whatsapp"),
                                        href="https://wa.me/6287860587871",
                                        target="_blank",
                                        className="social-icon",
                                    ),
                                ],
                                className="mt-3 mb-4",
                            ),

                            dbc.Button(
                                "Masuk ke Dashboard Analisis",
                                href="/analysis",
                                color="primary",
                                className="me-2 mt-3",
                                style={"padding": "14px 22px", "fontSize": "1.1rem"}
                            ),
                        ],
                        md=6,
                    ),

                    # RIGHT SIDE IMAGE
                    dbc.Col(
                        html.Img(
                            src="/assets/foto_gua.jpg",
                            style={
                                "width": "100%",
                                "borderRadius": "24px",
                                "boxShadow": "0 20px 40px rgba(0,0,0,0.5)",
                                "objectFit": "cover",
                            },
                        ),
                        md=6,
                    ),
                ],
                className="mt-5 align-items-center",
            ),
        ],
        fluid=True,
        className="about-container page-fade",
    )

# =====================================
# 2. PAGE 2 – DASHBOARD ANALISIS (PUNYAMU YANG LAMA)
# =====================================
def create_dashboard_layout():
    return html.Div(
        [
            # ini persis seperti app.layout lama kamu (setelah navbar)
            html.Div(
                [
                    dcc.Tabs(
                        id="main-tabs",
                        value="coop-tab",
                        children=[
                            dcc.Tab(
                                label="💰 Analisis Koperasi (Nonparametrik)",
                                value="coop-tab",
                            ),
                            dcc.Tab(
                                label="⏰ Time Series Analysis",
                                value="tsa-tab",
                            ),
                            dcc.Tab(
                                label="🧬 Pattern Recognition (Clustering)",
                                value="pattern-tab",
                            ),
                        ],
                        className="mt-2",
                    ),
                    html.Div(id="tabs-content", className="p-4"),
                ],
                className="custom-tabs mt-2 page-fade",
            ),
        ]
    )


# =====================================
# 3. APP LAYOUT – ROUTER
# =====================================
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        create_navbar(),                 # navbar selalu ada
        html.Div(id="page-content"),     # isi halaman diganti via callback
    ]
)


# =====================================
# 4. ROUTING ANTAR PAGE
# =====================================
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname: str):
    # landing / about
    if pathname in ("/", "/about", None):
        return create_about_layout()

    # dashboard analisis (page 2)
    if pathname in ("/dashboard", "/analysis", "/main"):
        return create_dashboard_layout()

    # 404 sederhana
    return html.Div(
        [
            html.H2("404 — Halaman tidak ditemukan", className="text-white"),
            html.P(f"Path: {pathname}", className="text-muted"),
            dbc.Button("Kembali ke Beranda", href="/", color="primary", className="mt-3"),
        ],
        className="p-5",
    )


# =====================================
# 5. CALLBACK TAB (PERSIS SAMA DENGAN PUNYAMU)
# =====================================
@callback(
    Output("tabs-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab_content(tab):
    if tab == "tsa-tab":
        return create_tsa_tab()
    elif tab == "coop-tab":
        return create_cooperatives_layout()
    elif tab == "pattern-tab":
        return create_pattern_layout()
    return html.Div("Pilih tab untuk melihat analisis.")


if __name__ == "__main__":
    app.run(debug=True, port=8050)
