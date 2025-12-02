# app.py
import dash # type: ignore
from dash import dcc, html, Input, Output, callback # type: ignore
import dash_bootstrap_components as dbc # type: ignore
import dash_leaflet   # type: ignore

from components.navbar import create_navbar
from components.time_series import create_tsa_tab
from components.cooperatives_analysis import create_cooperatives_layout
from components.pattern_recognition import create_pattern_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Economic Analysis Dashboard"
server = app.server

app.layout = html.Div(
    [
        create_navbar(),
        html.Div(
            [
                dcc.Tabs(
                    id="main-tabs",
                    value="coop-tab",
                    children=[
                        dcc.Tab(
                            label="💦 Analisis Koperasi (Nonparametrik)",
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
            className="container-fluid main-container",
        ),
    ]
)


@callback(
    Output("tabs-content", "children"),
    Input("main-tabs", "value")
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
