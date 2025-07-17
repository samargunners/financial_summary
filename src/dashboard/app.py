"""Main dashboard application."""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import logging
from typing import Dict, Any

from config.settings import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG
from src.dashboard.pages import overview, store_analysis, trends
from src.data_processing.database_manager import DatabaseManager


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Financial Analysis Dashboard"
)

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Financial Analysis Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Navigation
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Store Analysis", href="/store-analysis", active="exact")),
                dbc.NavItem(dbc.NavLink("Trends", href="/trends", active="exact")),
            ], pills=True, className="mb-3")
        ])
    ]),
    
    # Page content
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
    
], fluid=True)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """Display the appropriate page based on URL pathname."""
    if pathname == "/store-analysis":
        return store_analysis.layout
    elif pathname == "/trends":
        return trends.layout
    else:
        return overview.layout


def load_data() -> pd.DataFrame:
    """Load financial data from database."""
    try:
        db_manager = DatabaseManager()
        query = """
        SELECT date, amount, category, subcategory, description, 
               store_location, payment_method
        FROM financial_data
        ORDER BY date DESC
        """
        df = db_manager.query_data(query)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def get_summary_stats() -> Dict[str, Any]:
    """Get summary statistics for the dashboard."""
    try:
        df = load_data()
        if df.empty:
            return {
                'total_transactions': 0,
                'total_revenue': 0,
                'total_expenses': 0,
                'net_income': 0
            }
        
        # Separate revenue and expenses
        revenue = df[df['amount'] > 0]['amount'].sum()
        expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        return {
            'total_transactions': len(df),
            'total_revenue': revenue,
            'total_expenses': expenses,
            'net_income': revenue - expenses,
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            }
        }
    except Exception as e:
        logging.error(f"Error calculating summary stats: {str(e)}")
        return {}


# Register callbacks from page modules
overview.register_callbacks(app)
store_analysis.register_callbacks(app)
trends.register_callbacks(app)


def run_server(host: str = DASHBOARD_HOST, 
               port: int = DASHBOARD_PORT, 
               debug: bool = DASHBOARD_DEBUG):
    """Run the dashboard server."""
    logging.info(f"Starting dashboard server at http://{host}:{port}")
    app.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_server()
