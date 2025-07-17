"""Trends analysis page for the dashboard."""

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta

from src.dashboard.components.charts import ChartBuilder
from src.dashboard.components.filters import FilterBuilder
from src.data_processing.database_manager import DatabaseManager
from src.analysis.trend_analysis import TrendAnalyzer
from src.analysis.financial_metrics import FinancialMetricsCalculator


# Initialize components
chart_builder = ChartBuilder()
filter_builder = FilterBuilder()
trend_analyzer = TrendAnalyzer()
metrics_calculator = FinancialMetricsCalculator()

# Layout for the trends page
layout = dbc.Container([
    # Page header
    dbc.Row([
        dbc.Col([
            html.H2("Trend Analysis", className="mb-4"),
            html.P("Analyze financial trends, patterns, and forecasts", 
                  className="text-muted")
        ])
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trend Analysis Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Analysis Period"),
                            dcc.DatePickerRange(
                                id="trends-date-range",
                                start_date=datetime.now() - timedelta(days=180),
                                end_date=datetime.now(),
                                display_format='YYYY-MM-DD'
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Trend Type"),
                            dcc.Dropdown(
                                id="trend-type-filter",
                                options=[
                                    {'label': 'Revenue', 'value': 'revenue'},
                                    {'label': 'Transaction Count', 'value': 'count'},
                                    {'label': 'Average Transaction', 'value': 'average'}
                                ],
                                value='revenue',
                                clearable=False
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Aggregation Level"),
                            dcc.Dropdown(
                                id="aggregation-filter",
                                options=[
                                    {'label': 'Daily', 'value': 'D'},
                                    {'label': 'Weekly', 'value': 'W'},
                                    {'label': 'Monthly', 'value': 'M'},
                                    {'label': 'Quarterly', 'value': 'Q'}
                                ],
                                value='D',
                                clearable=False
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Category"),
                            dcc.Dropdown(
                                id="trends-category-filter",
                                placeholder="All categories",
                                multi=True
                            )
                        ], width=3)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Trend metrics cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Trend Direction", className="card-title"),
                    html.H3(id="trend-direction-card", className="text-primary"),
                    html.P(id="trend-strength-text", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Growth Rate", className="card-title"),
                    html.H3(id="growth-rate-card", className="text-success"),
                    html.P("Average monthly", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Volatility", className="card-title"),
                    html.H3(id="volatility-card", className="text-warning"),
                    html.P("Standard deviation", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("R-Squared", className="card-title"),
                    html.H3(id="r-squared-card", className="text-info"),
                    html.P("Trend fit quality", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Main trend charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Primary Trend Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="main-trend-chart")
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trend Decomposition"),
                dbc.CardBody([
                    dcc.Graph(id="trend-decomposition-chart")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Secondary analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Moving Averages"),
                dbc.CardBody([
                    dcc.Graph(id="moving-averages-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Volatility Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="volatility-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Seasonal and cyclical analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Seasonal Patterns"),
                dbc.CardBody([
                    dcc.Graph(id="seasonal-patterns-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Anomaly Detection"),
                dbc.CardBody([
                    dcc.Graph(id="anomaly-detection-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Forecast section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Trend Forecast", className="mb-0")
                        ], width=8),
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Forecast Days:"),
                                dbc.Input(id="forecast-days", type="number", value=30, min=1, max=365)
                            ], size="sm")
                        ], width=4)
                    ])
                ]),
                dbc.CardBody([
                    dcc.Graph(id="forecast-chart")
                ])
            ])
        ])
    ])
], fluid=True)


def register_callbacks(app):
    """Register callbacks for the trends page."""
    
    @app.callback(
        Output("trends-category-filter", "options"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date")]
    )
    def update_trends_filter_options(start_date, end_date):
        """Update filter options based on available data."""
        try:
            df = load_trends_data(start_date, end_date)
            
            if df.empty:
                return []
            
            category_options = filter_builder.get_category_options(df)
            return category_options
            
        except Exception as e:
            print(f"Error updating trends filter options: {e}")
            return []
    
    @app.callback(
        [Output("trend-direction-card", "children"),
         Output("trend-strength-text", "children"),
         Output("growth-rate-card", "children"),
         Output("volatility-card", "children"),
         Output("r-squared-card", "children")],
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("trend-type-filter", "value"),
         Input("aggregation-filter", "value"),
         Input("trends-category-filter", "value")]
    )
    def update_trend_metrics(start_date, end_date, trend_type, aggregation, categories):
        """Update trend metrics cards."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return "N/A", "No data", "0%", "0", "0"
            
            # Aggregate data based on selected level
            df_agg = aggregate_data(df, aggregation, trend_type)
            
            if df_agg.empty:
                return "N/A", "No data", "0%", "0", "0"
            
            # Perform trend analysis
            trend_result = trend_analyzer.detect_trend(df_agg, 'value', 'period')
            
            # Calculate growth rate
            growth_metrics = metrics_calculator.calculate_growth_metrics(df, 'amount', 'date', 'M')
            avg_growth = growth_metrics['amount_growth_rate'].mean() if not growth_metrics.empty else 0
            
            # Calculate volatility
            volatility_df = trend_analyzer.calculate_volatility(df, 'amount', 'date')
            avg_volatility = volatility_df['volatility'].mean() if not volatility_df.empty else 0
            
            return (
                trend_result.trend_direction.title(),
                f"Strength: {trend_result.trend_strength:.2f}",
                f"{avg_growth:.1f}%",
                f"{avg_volatility:.2f}",
                f"{trend_result.r_squared:.3f}"
            )
            
        except Exception as e:
            print(f"Error updating trend metrics: {e}")
            return "Error", "Error", "0%", "0", "0"
    
    @app.callback(
        Output("main-trend-chart", "figure"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("trend-type-filter", "value"),
         Input("aggregation-filter", "value"),
         Input("trends-category-filter", "value")]
    )
    def update_main_trend_chart(start_date, end_date, trend_type, aggregation, categories):
        """Update main trend chart."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return chart_builder._create_empty_chart("No data available")
            
            df_agg = aggregate_data(df, aggregation, trend_type)
            
            if df_agg.empty:
                return chart_builder._create_empty_chart("No aggregated data")
            
            return chart_builder.create_revenue_trend_chart(df_agg, 'period', 'value')
            
        except Exception as e:
            print(f"Error updating main trend chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("moving-averages-chart", "figure"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("trend-type-filter", "value"),
         Input("trends-category-filter", "value")]
    )
    def update_moving_averages_chart(start_date, end_date, trend_type, categories):
        """Update moving averages chart."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return chart_builder._create_empty_chart("No data available")
            
            # Calculate moving averages
            df_ma = trend_analyzer.analyze_moving_averages(df, 'amount', 'date', [7, 30, 90])
            
            return chart_builder.create_multi_line_chart(
                df_ma,
                'date',
                ['amount', 'ma_7', 'ma_30', 'ma_90'],
                'Moving Averages Analysis'
            )
            
        except Exception as e:
            print(f"Error updating moving averages chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("volatility-chart", "figure"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("trends-category-filter", "value")]
    )
    def update_volatility_chart(start_date, end_date, categories):
        """Update volatility chart."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return chart_builder._create_empty_chart("No data available")
            
            # Calculate volatility
            df_vol = trend_analyzer.calculate_volatility(df, 'amount', 'date', 30)
            
            return chart_builder.create_multi_line_chart(
                df_vol,
                'date',
                ['volatility', 'avg_volatility'],
                'Volatility Analysis'
            )
            
        except Exception as e:
            print(f"Error updating volatility chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("seasonal-patterns-chart", "figure"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("trends-category-filter", "value")]
    )
    def update_seasonal_patterns_chart(start_date, end_date, categories):
        """Update seasonal patterns chart."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return chart_builder._create_empty_chart("No data available")
            
            # Calculate seasonal metrics
            seasonal_data = trend_analyzer.analyze_cycles(df, 'amount', 'date')
            
            if not seasonal_data.get('monthly_patterns'):
                return chart_builder._create_empty_chart("No seasonal data")
            
            # Create monthly patterns chart
            monthly_df = pd.DataFrame(seasonal_data['monthly_patterns']).T
            monthly_df = monthly_df.reset_index()
            monthly_df.columns = ['month'] + list(monthly_df.columns[1:])
            
            return chart_builder.create_bar_chart(
                monthly_df,
                'month',
                'mean',
                'Monthly Seasonal Patterns'
            )
            
        except Exception as e:
            print(f"Error updating seasonal patterns chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("anomaly-detection-chart", "figure"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("trends-category-filter", "value")]
    )
    def update_anomaly_detection_chart(start_date, end_date, categories):
        """Update anomaly detection chart."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return chart_builder._create_empty_chart("No data available")
            
            # Detect anomalies
            df_anomalies = trend_analyzer.detect_anomalies(df, 'amount', 'iqr', 1.5)
            
            return chart_builder.create_scatter_plot(
                df_anomalies,
                'date',
                'amount',
                'Anomaly Detection',
                color_col='is_anomaly'
            )
            
        except Exception as e:
            print(f"Error updating anomaly detection chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")
    
    @app.callback(
        Output("forecast-chart", "figure"),
        [Input("trends-date-range", "start_date"),
         Input("trends-date-range", "end_date"),
         Input("forecast-days", "value"),
         Input("trends-category-filter", "value")]
    )
    def update_forecast_chart(start_date, end_date, forecast_days, categories):
        """Update forecast chart."""
        try:
            df = load_trends_data(start_date, end_date)
            df = filter_trends_data(df, categories)
            
            if df.empty:
                return chart_builder._create_empty_chart("No data available")
            
            # Generate forecast
            forecast_df = trend_analyzer.forecast_trend(df, 'amount', 'date', forecast_days or 30)
            
            if forecast_df.empty:
                return chart_builder._create_empty_chart("Unable to generate forecast")
            
            # Combine historical and forecast data
            historical_chart = chart_builder.create_revenue_trend_chart(df, 'date', 'amount')
            
            # Add forecast line
            historical_chart.add_scatter(
                x=forecast_df['date'],
                y=forecast_df['amount_forecast'],
                mode='lines',
                name='Forecast',
                line=dict(dash='dash', color='red', width=2)
            )
            
            historical_chart.update_layout(title="Historical Data and Trend Forecast")
            
            return historical_chart
            
        except Exception as e:
            print(f"Error updating forecast chart: {e}")
            return chart_builder._create_empty_chart("Error loading data")


def load_trends_data(start_date=None, end_date=None):
    """Load data for trend analysis."""
    try:
        db_manager = DatabaseManager()
        
        base_query = """
        SELECT date, amount, category, subcategory, description, 
               store_location, payment_method
        FROM financial_data
        """
        
        if start_date and end_date:
            query = f"{base_query} WHERE date BETWEEN ? AND ? ORDER BY date ASC"
            params = (start_date, end_date)
        else:
            query = f"{base_query} ORDER BY date ASC LIMIT 1000"
            params = None
        
        df = db_manager.query_data(query, params)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        print(f"Error loading trends data: {e}")
        return pd.DataFrame()


def filter_trends_data(df, categories=None):
    """Apply filters to trends data."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply category filter
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    return filtered_df


def aggregate_data(df, aggregation, trend_type):
    """Aggregate data based on selected parameters."""
    if df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['period'] = df_copy['date'].dt.to_period(aggregation)
    
    if trend_type == 'revenue':
        agg_df = df_copy.groupby('period')['amount'].sum().reset_index()
    elif trend_type == 'count':
        agg_df = df_copy.groupby('period')['amount'].count().reset_index()
    elif trend_type == 'average':
        agg_df = df_copy.groupby('period')['amount'].mean().reset_index()
    else:
        agg_df = df_copy.groupby('period')['amount'].sum().reset_index()
    
    agg_df.columns = ['period', 'value']
    agg_df['period'] = agg_df['period'].dt.to_timestamp()
    
    return agg_df
