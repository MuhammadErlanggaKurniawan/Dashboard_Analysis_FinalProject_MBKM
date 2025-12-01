from dash import html
import dash_bootstrap_components as dbc

def create_navbar():
    return dbc.NavbarSimple(
        brand="Dashboard Analysis",
        color="primary",
        dark=True,
        # children dikosongin biar ga ada tab dummy
        children=[]  
    )