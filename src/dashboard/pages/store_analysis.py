"""Store analysis page for the dashboard."""

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta

from src.dashboard.components.charts import ChartBuilder
from src.dashboard.components.filters import FilterBuilder
from src.data_processing.database_manager import DatabaseManager
from src.analysis.financial_metrics import FinancialMetricsCalculator


# Initialize components
chart_builder = ChartBuilder()
filter_builder = FilterBuilder()
metrics_calculator = FinancialMetricsCalculator()

# Layout for the store analysis page
layout = dbc.Container([
    # Page header
    dbc.Row([
        dbc.Col([
            html.H2("Store Analysis", className="mb-4"),
            html.P("Analyze performance across different store locations", 
                  className="text-muted")
        ])
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Store Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="store-date-range",
                                start_date=datetime.now() - timedelta(days=90),
                                end_date=datetime.now(),
                                display_format='YYYY-MM-DD'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Store Locations"),
                            dcc.Dropdown(
                                id="store-location-filter",
                                placeholder="Select stores...",
                                multi=True
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Category"),
                            dcc.Dropdown(
                                id="store-category-filter",
                                placeholder="Select categories...",
                                multi=True
                            )
                        ], width=4)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Store performance cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Top Performing Store", className="card-title"),
                    html.H3(id="top-store-card", className="text-success"),
                    html.P(id="top-store-revenue", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Stores", className="card-title"),
                    html.H3(id="total-stores-card", className="text-primary"),
                    html.P("Active locations", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Revenue/Store", className="card-title"),
                    html.H3(id="avg-revenue-store-card", className="text-info"),
                    html.P("Per store average", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Store Efficiency", className="card-title"),
                    html.H3(id="store-efficiency-card", className="text-warning"),
                    html.P("Revenue per transaction", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Main charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Store Performance Comparison"),
                dbc.CardBody([
                    dcc.Graph(id="store-performance-chart")
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Store Revenue Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="store-revenue-distribution")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Additional analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Store Performance Heatmap"),
                dbc.CardBody([
                    dcc.Graph(id="store-heatmap")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Store Trend Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="store-trends")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Store details table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Store Performance Details"),
                dbc.CardBody([
                    html.Div(id="store-details-table")
                ])
            ])
        ])
    ])
], fluid=True)


def register_callbacks(app):
    """Register callbacks for the store analysis page."""
    
    @app.callback(
        [Output("store-location-filter", "options"),
         Output("store-category-filter", "options")],
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date")]
    )
    def update_store_filter_options(start_date, end_date):
        """Update filter options based on available data."""
        try:
            df = load_store_data(start_date, end_date)
            
            if df.empty:
                return [], []
            
            # Store location options
            store_options = filter_builder.get_store_options(df)
            
            # Category options
            category_options = filter_builder.get_category_options(df)
            
            return store_options, category_options
            
        except Exception as e:
            print(f"Error updating store filter options: {e}")
            return [], []
    
    @app.callback(
        [Output("top-store-card", "children"),
         Output("top-store-revenue", "children"),
         Output("total-stores-card", "children"),
         Output("avg-revenue-store-card", "children"),
         Output("store-efficiency-card", "children")],
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date"),
         Input("store-location-filter", "value"),
         Input("store-category-filter", "value")]
    )
    def update_store_cards(start_date, end_date, stores, categories):
        """Update store performance cards."""
        try:
            df = load_store_data(start_date, end_date)
            df = filter_store_data(df, stores, categories)
            
            if df.empty or 'store_location' not in df.columns:
                return "N/A", "No data", "0", "$0", "$0"
            
            # Calculate store metrics
            store_metrics = df.groupby('store_location')['amount'].agg(['sum', 'count', 'mean']).reset_index()
            store_metrics.columns = ['store', 'total_revenue', 'transaction_count', 'avg_transaction']
            store_metrics = store_metrics.sort_values('total_revenue', ascending=False)
            
            if store_metrics.empty:
                return "N/A", "No data", "0", "$0", "$0"
            
            # Top performing store
            top_store = store_metrics.iloc[0]
            top_store_name = top_store['store']
            top_store_revenue = f"${top_store['total_revenue']:,.2f}"
            
            # Total stores
            total_stores = len(store_metrics)
            
            # Average revenue per store
            avg_revenue = store_metrics['total_revenue'].mean()
            
            # Store efficiency (avg transaction amount)
            avg_efficiency = store_metrics['avg_transaction'].mean()
            
            return (
                top_store_name,
                top_store_revenue,
                str(total_stores),
                f"${avg_revenue:,.2f}",
                f"${avg_efficiency:,.2f}"
            )
            
        except Exception as e:
            print(f"Error updating store cards: {e}")
            return "Error", "Error", "0", "$0", "$0"
    
    @app.callback(
        Output("store-performance-chart", "figure"),
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date"),
         Input("store-location-filter", "value"),
         Input("store-category-filter", "value")]
    )
    def update_store_performance_chart(start_date, end_date, stores, categories):
        """Update store performance comparison chart."""
        try:
            df = load_store_data(start_date, end_date)
            df = filter_store_data(df, stores, categories)
            
            if df.empty or 'store_location' not in df.columns:
                return chart_builder._create_empty_chart("No store data available")
            
            store_metrics = df.groupby('store_location')['amount'].agg(['sum', 'count']).reset_index()
            store_metrics.columns = ['store', 'revenue', 'transactions']
            store_metrics = store_metrics.sort_values('revenue', ascending=True)
            
            return chart_builder.create_bar_chart(
                store_metrics,
                'store',
                'revenue',
                'Store Revenue Comparison'
            )
            
        except Exception as e:
            print(f"Error updating store performance chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("store-revenue-distribution", "figure"),
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date"),
         Input("store-location-filter", "value"),
         Input("store-category-filter", "value")]
    )
    def update_store_revenue_distribution(start_date, end_date, stores, categories):
        """Update store revenue distribution chart."""
        try:
            df = load_store_data(start_date, end_date)
            df = filter_store_data(df, stores, categories)
            
            if df.empty or 'store_location' not in df.columns:
                return chart_builder._create_empty_chart("No store data available")
            
            return chart_builder.create_category_pie_chart(df, 'store_location', 'amount')
            
        except Exception as e:
            print(f"Error updating store revenue distribution: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("store-heatmap", "figure"),
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date"),
         Input("store-location-filter", "value"),
         Input("store-category-filter", "value")]
    )
    def update_store_heatmap(start_date, end_date, stores, categories):
        """Update store performance heatmap."""
        try:
            df = load_store_data(start_date, end_date)
            df = filter_store_data(df, stores, categories)
            
            if df.empty or 'store_location' not in df.columns:
                return chart_builder._create_empty_chart("No store data available")
            
            # Create month-store heatmap
            df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
            
            return chart_builder.create_heatmap(
                df,
                'month',
                'store_location',
                'amount',
                'Store Performance by Month'
            )
            
        except Exception as e:
            print(f"Error updating store heatmap: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("store-trends", "figure"),
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date"),
         Input("store-location-filter", "value"),
         Input("store-category-filter", "value")]
    )
    def update_store_trends(start_date, end_date, stores, categories):
        """Update store trend analysis chart."""
        try:
            df = load_store_data(start_date, end_date)
            df = filter_store_data(df, stores, categories)
            
            if df.empty or 'store_location' not in df.columns:
                return chart_builder._create_empty_chart("No store data available")
            
            # Get top 5 stores by revenue
            top_stores = df.groupby('store_location')['amount'].sum().nlargest(5).index
            df_top = df[df['store_location'].isin(top_stores)]
            
            # Create daily revenue trends for top stores
            daily_trends = df_top.groupby(['date', 'store_location'])['amount'].sum().reset_index()
            daily_trends = daily_trends.pivot(index='date', columns='store_location', values='amount').fillna(0)
            
            return chart_builder.create_multi_line_chart(
                daily_trends.reset_index(),
                'date',
                list(daily_trends.columns),
                'Top Store Revenue Trends'
            )
            
        except Exception as e:
            print(f"Error updating store trends: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("store-details-table", "children"),
        [Input("store-date-range", "start_date"),
         Input("store-date-range", "end_date"),
         Input("store-location-filter", "value"),
         Input("store-category-filter", "value")]
    )
    def update_store_details_table(start_date, end_date, stores, categories):
        """Update store details table."""
        try:
            df = load_store_data(start_date, end_date)
            df = filter_store_data(df, stores, categories)
            
            if df.empty or 'store_location' not in df.columns:
                return html.P("No store data available")
            
            # Calculate detailed store metrics
            store_details = df.groupby('store_location').agg({
                'amount': ['sum', 'count', 'mean', 'std'],
                'date': ['min', 'max']
            }).round(2)
            
            store_details.columns = ['Total Revenue', 'Transactions', 'Avg Transaction', 
                                   'Std Dev', 'First Sale', 'Last Sale']
            store_details = store_details.reset_index()
            
            # Format currency columns
            for col in ['Total Revenue', 'Avg Transaction', 'Std Dev']:
                store_details[col] = store_details[col].apply(lambda x: f"${x:,.2f}")
            
            # Create table
            table = dbc.Table.from_dataframe(
                store_details,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
            
            return table
            
        except Exception as e:
            print(f"Error updating store details table: {e}")
            return html.P("Error loading store details")


def load_store_data(start_date=None, end_date=None):
    """Load data for store analysis."""
    try:
        db_manager = DatabaseManager()
        
        base_query = """
        SELECT date, amount, category, subcategory, description, 
               store_location, payment_method
        FROM financial_data
        WHERE store_location IS NOT NULL AND store_location != ''
        """
        
        if start_date and end_date:
            query = f"{base_query} AND date BETWEEN ? AND ? ORDER BY date DESC"
            params = (start_date, end_date)
        else:
            query = f"{base_query} ORDER BY date DESC LIMIT 1000"
            params = None
        
        df = db_manager.query_data(query, params)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        print(f"Error loading store data: {e}")
        return pd.DataFrame()


def filter_store_data(df, stores=None, categories=None):
    """Apply filters to store data."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply store filter
    if stores and len(stores) > 0:
        filtered_df = filtered_df[filtered_df['store_location'].isin(stores)]
    
    # Apply category filter
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    return filtered_df
