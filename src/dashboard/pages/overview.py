"""Overview page for the dashboard."""

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.dashboard.components.charts import ChartBuilder
from src.dashboard.components.filters import FilterBuilder
from src.data_processing.database_manager import DatabaseManager
from src.analysis.financial_metrics import FinancialMetricsCalculator


# Initialize components
chart_builder = ChartBuilder()
filter_builder = FilterBuilder()
metrics_calculator = FinancialMetricsCalculator()

# Layout for the overview page
layout = dbc.Container([
    # Summary cards row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Revenue", className="card-title"),
                    html.H2(id="total-revenue-card", className="text-success"),
                    html.P("Current period", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Expenses", className="card-title"),
                    html.H2(id="total-expenses-card", className="text-danger"),
                    html.P("Current period", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Net Income", className="card-title"),
                    html.H2(id="net-income-card", className="text-primary"),
                    html.P("Current period", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Transactions", className="card-title"),
                    html.H2(id="transaction-count-card", className="text-info"),
                    html.P("Total count", className="card-text")
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    # Filters row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="overview-date-range",
                                start_date=datetime.now() - timedelta(days=30),
                                end_date=datetime.now(),
                                display_format='YYYY-MM-DD'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Category"),
                            dcc.Dropdown(
                                id="overview-category-filter",
                                placeholder="Select categories...",
                                multi=True
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Amount Range"),
                            dcc.RangeSlider(
                                id="overview-amount-range",
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Charts row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Revenue Trend"),
                dbc.CardBody([
                    dcc.Graph(id="overview-revenue-trend")
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Category Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="overview-category-pie")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Additional charts row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Monthly Performance"),
                dbc.CardBody([
                    dcc.Graph(id="overview-monthly-performance")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Payment Method Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="overview-payment-methods")
                ])
            ])
        ], width=6)
    ])
], fluid=True)


def register_callbacks(app):
    """Register callbacks for the overview page."""
    
    @app.callback(
        [Output("overview-category-filter", "options"),
         Output("overview-amount-range", "min"),
         Output("overview-amount-range", "max"),
         Output("overview-amount-range", "value")],
        [Input("overview-date-range", "start_date"),
         Input("overview-date-range", "end_date")]
    )
    def update_filter_options(start_date, end_date):
        """Update filter options based on available data."""
        try:
            df = load_overview_data(start_date, end_date)
            
            if df.empty:
                return [], 0, 100, [0, 100]
            
            # Category options
            category_options = filter_builder.get_category_options(df)
            
            # Amount range
            amount_range = filter_builder.get_amount_range(df)
            
            return (
                category_options,
                amount_range['min'],
                amount_range['max'],
                [amount_range['min'], amount_range['max']]
            )
            
        except Exception as e:
            print(f"Error updating filter options: {e}")
            return [], 0, 100, [0, 100]
    
    @app.callback(
        [Output("total-revenue-card", "children"),
         Output("total-expenses-card", "children"),
         Output("net-income-card", "children"),
         Output("transaction-count-card", "children")],
        [Input("overview-date-range", "start_date"),
         Input("overview-date-range", "end_date"),
         Input("overview-category-filter", "value"),
         Input("overview-amount-range", "value")]
    )
    def update_summary_cards(start_date, end_date, categories, amount_range):
        """Update summary metric cards."""
        try:
            df = load_overview_data(start_date, end_date)
            df = filter_data(df, categories, amount_range)
            
            if df.empty:
                return "$0", "$0", "$0", "0"
            
            metrics = metrics_calculator.calculate_basic_metrics(df)
            
            return (
                f"${metrics.total_revenue:,.2f}",
                f"${metrics.total_expenses:,.2f}",
                f"${metrics.net_income:,.2f}",
                f"{metrics.transaction_count:,}"
            )
            
        except Exception as e:
            print(f"Error updating summary cards: {e}")
            return "$0", "$0", "$0", "0"
    
    @app.callback(
        Output("overview-revenue-trend", "figure"),
        [Input("overview-date-range", "start_date"),
         Input("overview-date-range", "end_date"),
         Input("overview-category-filter", "value"),
         Input("overview-amount-range", "value")]
    )
    def update_revenue_trend(start_date, end_date, categories, amount_range):
        """Update revenue trend chart."""
        try:
            df = load_overview_data(start_date, end_date)
            df = filter_data(df, categories, amount_range)
            
            return chart_builder.create_revenue_trend_chart(df)
            
        except Exception as e:
            print(f"Error updating revenue trend: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("overview-category-pie", "figure"),
        [Input("overview-date-range", "start_date"),
         Input("overview-date-range", "end_date"),
         Input("overview-category-filter", "value"),
         Input("overview-amount-range", "value")]
    )
    def update_category_pie(start_date, end_date, categories, amount_range):
        """Update category pie chart."""
        try:
            df = load_overview_data(start_date, end_date)
            df = filter_data(df, categories, amount_range)
            
            return chart_builder.create_category_pie_chart(df)
            
        except Exception as e:
            print(f"Error updating category pie: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("overview-monthly-performance", "figure"),
        [Input("overview-date-range", "start_date"),
         Input("overview-date-range", "end_date"),
         Input("overview-category-filter", "value"),
         Input("overview-amount-range", "value")]
    )
    def update_monthly_performance(start_date, end_date, categories, amount_range):
        """Update monthly performance chart."""
        try:
            df = load_overview_data(start_date, end_date)
            df = filter_data(df, categories, amount_range)
            
            monthly_metrics = metrics_calculator.calculate_monthly_metrics(df)
            
            if monthly_metrics.empty:
                return chart_builder._create_empty_chart("No data available")
            
            return chart_builder.create_multi_line_chart(
                monthly_metrics,
                'period',
                ['total_revenue', 'total_expenses', 'net_income'],
                'Monthly Financial Performance'
            )
            
        except Exception as e:
            print(f"Error updating monthly performance: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("overview-payment-methods", "figure"),
        [Input("overview-date-range", "start_date"),
         Input("overview-date-range", "end_date"),
         Input("overview-category-filter", "value"),
         Input("overview-amount-range", "value")]
    )
    def update_payment_methods(start_date, end_date, categories, amount_range):
        """Update payment methods chart."""
        try:
            df = load_overview_data(start_date, end_date)
            df = filter_data(df, categories, amount_range)
            
            if df.empty or 'payment_method' not in df.columns:
                return chart_builder._create_empty_chart("No payment method data")
            
            payment_summary = df.groupby('payment_method')['amount'].sum().reset_index()
            
            return chart_builder.create_bar_chart(
                payment_summary,
                'payment_method',
                'amount',
                'Revenue by Payment Method'
            )
            
        except Exception as e:
            print(f"Error updating payment methods: {e}")
            return chart_builder._create_empty_chart("Error loading data")


def load_overview_data(start_date=None, end_date=None):
    """Load data for the overview page."""
    try:
        db_manager = DatabaseManager()
        
        base_query = """
        SELECT date, amount, category, subcategory, description, 
               store_location, payment_method
        FROM financial_data
        """
        
        if start_date and end_date:
            query = f"{base_query} WHERE date BETWEEN ? AND ? ORDER BY date DESC"
            params = (start_date, end_date)
        else:
            query = f"{base_query} ORDER BY date DESC LIMIT 1000"
            params = None
        
        df = db_manager.query_data(query, params)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        print(f"Error loading overview data: {e}")
        return pd.DataFrame()


def filter_data(df, categories=None, amount_range=None):
    """Apply filters to the dataframe."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply category filter
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    # Apply amount range filter
    if amount_range and len(amount_range) == 2:
        min_amount, max_amount = amount_range
        filtered_df = filtered_df[
            (filtered_df['amount'] >= min_amount) & 
            (filtered_df['amount'] <= max_amount)
        ]
    
    return filtered_df
