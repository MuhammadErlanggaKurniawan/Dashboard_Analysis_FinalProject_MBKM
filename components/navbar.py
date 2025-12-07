from dash import html
import dash_bootstrap_components as dbc

def create_navbar():
    return dbc.NavbarSimple(
        brand="📊 Economic & Cooperative Analytics Dashboard – Jawa Timur",
        brand_href="/",
        color="primary",
        dark=True,
        fluid=True,
        children=[ dbc.Button("About", href="/", outline=True, color="light", className="me-2"),]
    )