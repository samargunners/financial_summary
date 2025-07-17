"""Trend analysis functionality."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrendResult:
    """Data class for storing trend analysis results."""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1 scale
    slope: float
    r_squared: float
    prediction_accuracy: float
    seasonal_component: bool


class TrendAnalyzer:
    """Analyze trends in financial data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_trend(self, df: pd.DataFrame,
                    value_col: str = 'amount',
                    date_col: str = 'date',
                    method: str = 'linear') -> TrendResult:
        """Detect trend in time series data.
        
        Args:
            df: DataFrame with time series data
            value_col: Name of value column
            date_col: Name of date column
            method: Trend detection method ('linear', 'polynomial')
            
        Returns:
            TrendResult object with trend analysis
        """
        if df.empty or len(df) < 3:
            return TrendResult('stable', 0.0, 0.0, 0.0, 0.0, False)
        
        # Prepare data
        df_sorted = df.sort_values(date_col).copy()
        df_sorted['date_numeric'] = pd.to_datetime(df_sorted[date_col]).astype(np.int64)
        
        x = df_sorted['date_numeric'].values
        y = df_sorted[value_col].values
        
        # Normalize x for better numerical stability
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(x)
        
        try:
            if method == 'linear':
                # Linear regression
                coeffs = np.polyfit(x_norm, y, 1)
                slope = coeffs[0]
                
                # Calculate R-squared
                y_pred = np.polyval(coeffs, x_norm)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
            elif method == 'polynomial':
                # Polynomial regression (degree 2)
                coeffs = np.polyfit(x_norm, y, 2)
                slope = coeffs[1]  # Linear coefficient
                
                y_pred = np.polyval(coeffs, x_norm)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            if abs(slope) < 0.1 * np.std(y):
                trend_direction = 'stable'
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = 'increasing'
                trend_strength = min(abs(slope) / np.std(y), 1.0)
            else:
                trend_direction = 'decreasing'
                trend_strength = min(abs(slope) / np.std(y), 1.0)
            
            # Calculate prediction accuracy (MAE)
            mae = np.mean(np.abs(y - y_pred))
            prediction_accuracy = max(0, 1 - (mae / np.std(y))) if np.std(y) != 0 else 0
            
            # Check for seasonal component (simplified)
            seasonal_component = self._detect_seasonality(df_sorted, value_col, date_col)
            
            return TrendResult(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                prediction_accuracy=prediction_accuracy,
                seasonal_component=seasonal_component
            )
            
        except Exception as e:
            self.logger.error(f"Error in trend detection: {str(e)}")
            return TrendResult('stable', 0.0, 0.0, 0.0, 0.0, False)
    
    def _detect_seasonality(self, df: pd.DataFrame,
                          value_col: str,
                          date_col: str) -> bool:
        """Detect if data has seasonal patterns.
        
        Args:
            df: DataFrame with time series data
            value_col: Name of value column
            date_col: Name of date column
            
        Returns:
            True if seasonal patterns detected
        """
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            
            # Group by month and calculate variance
            monthly_means = df_copy.groupby(df_copy[date_col].dt.month)[value_col].mean()
            
            if len(monthly_means) < 3:
                return False
            
            # If monthly variance is significantly higher than expected, assume seasonality
            monthly_variance = monthly_means.var()
            overall_variance = df_copy[value_col].var()
            
            return monthly_variance > 0.1 * overall_variance
            
        except Exception:
            return False
    
    def analyze_moving_averages(self, df: pd.DataFrame,
                              value_col: str = 'amount',
                              date_col: str = 'date',
                              windows: List[int] = None) -> pd.DataFrame:
        """Calculate moving averages for trend analysis.
        
        Args:
            df: DataFrame with time series data
            value_col: Name of value column
            date_col: Name of date column
            windows: List of window sizes for moving averages
            
        Returns:
            DataFrame with moving averages
        """
        if windows is None:
            windows = [7, 30, 90]  # 7-day, 30-day, 90-day
        
        df_sorted = df.sort_values(date_col).copy()
        
        for window in windows:
            if len(df_sorted) >= window:
                df_sorted[f'ma_{window}'] = df_sorted[value_col].rolling(window=window, min_periods=1).mean()
                df_sorted[f'ma_{window}_std'] = df_sorted[value_col].rolling(window=window, min_periods=1).std()
        
        return df_sorted
    
    def detect_anomalies(self, df: pd.DataFrame,
                        value_col: str = 'amount',
                        method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """Detect anomalies in financial data.
        
        Args:
            df: DataFrame with financial data
            value_col: Name of value column
            method: Anomaly detection method ('iqr', 'zscore')
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        df_copy = df.copy()
        
        if method == 'iqr':
            Q1 = df_copy[value_col].quantile(0.25)
            Q3 = df_copy[value_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_copy['is_anomaly'] = (df_copy[value_col] < lower_bound) | (df_copy[value_col] > upper_bound)
            df_copy['anomaly_score'] = np.where(
                df_copy[value_col] < lower_bound,
                (lower_bound - df_copy[value_col]) / IQR,
                np.where(
                    df_copy[value_col] > upper_bound,
                    (df_copy[value_col] - upper_bound) / IQR,
                    0
                )
            )
            
        elif method == 'zscore':
            mean_val = df_copy[value_col].mean()
            std_val = df_copy[value_col].std()
            
            if std_val == 0:
                df_copy['is_anomaly'] = False
                df_copy['anomaly_score'] = 0
            else:
                z_scores = np.abs((df_copy[value_col] - mean_val) / std_val)
                df_copy['is_anomaly'] = z_scores > threshold
                df_copy['anomaly_score'] = z_scores
        
        return df_copy
    
    def calculate_volatility(self, df: pd.DataFrame,
                           value_col: str = 'amount',
                           date_col: str = 'date',
                           window: int = 30) -> pd.DataFrame:
        """Calculate volatility metrics.
        
        Args:
            df: DataFrame with financial data
            value_col: Name of value column
            date_col: Name of date column
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility metrics
        """
        df_sorted = df.sort_values(date_col).copy()
        
        # Calculate returns (percentage change)
        df_sorted['returns'] = df_sorted[value_col].pct_change()
        
        # Rolling volatility (standard deviation of returns)
        df_sorted['volatility'] = df_sorted['returns'].rolling(window=window, min_periods=1).std()
        
        # Rolling variance
        df_sorted['variance'] = df_sorted['returns'].rolling(window=window, min_periods=1).var()
        
        # Average volatility
        df_sorted['avg_volatility'] = df_sorted['volatility'].expanding().mean()
        
        return df_sorted
    
    def forecast_trend(self, df: pd.DataFrame,
                      value_col: str = 'amount',
                      date_col: str = 'date',
                      periods: int = 30) -> pd.DataFrame:
        """Simple trend forecasting using linear regression.
        
        Args:
            df: DataFrame with historical data
            value_col: Name of value column
            date_col: Name of date column
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecast values
        """
        if df.empty or len(df) < 3:
            return pd.DataFrame()
        
        df_sorted = df.sort_values(date_col).copy()
        df_sorted['date_numeric'] = pd.to_datetime(df_sorted[date_col]).astype(np.int64)
        
        # Fit linear trend
        x = df_sorted['date_numeric'].values
        y = df_sorted[value_col].values
        
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(x)
        
        try:
            coeffs = np.polyfit(x_norm, y, 1)
            
            # Generate future dates
            last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])
            date_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
            
            # Convert to numeric and normalize
            future_x = date_range.astype(np.int64).values
            future_x_norm = (future_x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(future_x)
            
            # Predict
            future_y = np.polyval(coeffs, future_x_norm)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                date_col: date_range,
                f'{value_col}_forecast': future_y,
                'forecast_type': 'linear_trend'
            })
            
            return forecast_df
            
        except Exception as e:
            self.logger.error(f"Error in forecasting: {str(e)}")
            return pd.DataFrame()
    
    def analyze_cycles(self, df: pd.DataFrame,
                      value_col: str = 'amount',
                      date_col: str = 'date') -> Dict[str, Any]:
        """Analyze cyclical patterns in data.
        
        Args:
            df: DataFrame with time series data
            value_col: Name of value column
            date_col: Name of date column
            
        Returns:
            Dictionary with cycle analysis results
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        results = {
            'daily_patterns': {},
            'weekly_patterns': {},
            'monthly_patterns': {},
            'quarterly_patterns': {}
        }
        
        try:
            # Daily patterns (hour of day if available)
            if 'hour' in df_copy.columns or df_copy[date_col].dt.hour.nunique() > 1:
                hourly = df_copy.groupby(df_copy[date_col].dt.hour)[value_col].agg(['mean', 'std', 'count'])
                results['daily_patterns'] = hourly.to_dict('index')
            
            # Weekly patterns (day of week)
            weekly = df_copy.groupby(df_copy[date_col].dt.day_name())[value_col].agg(['mean', 'std', 'count'])
            results['weekly_patterns'] = weekly.to_dict('index')
            
            # Monthly patterns
            monthly = df_copy.groupby(df_copy[date_col].dt.month)[value_col].agg(['mean', 'std', 'count'])
            results['monthly_patterns'] = monthly.to_dict('index')
            
            # Quarterly patterns
            quarterly = df_copy.groupby(df_copy[date_col].dt.quarter)[value_col].agg(['mean', 'std', 'count'])
            results['quarterly_patterns'] = quarterly.to_dict('index')
            
        except Exception as e:
            self.logger.error(f"Error in cycle analysis: {str(e)}")
        
        return results
    
    def compare_periods(self, df: pd.DataFrame,
                       period1_start: str,
                       period1_end: str,
                       period2_start: str,
                       period2_end: str,
                       value_col: str = 'amount',
                       date_col: str = 'date') -> Dict[str, Any]:
        """Compare financial performance between two periods.
        
        Args:
            df: DataFrame with financial data
            value_col: Name of value column
            date_col: Name of date column
            period1_start: Start date for first period
            period1_end: End date for first period
            period2_start: Start date for second period
            period2_end: End date for second period
            
        Returns:
            Dictionary with comparison results
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Filter periods
        period1_data = df_copy[
            (df_copy[date_col] >= period1_start) & 
            (df_copy[date_col] <= period1_end)
        ]
        
        period2_data = df_copy[
            (df_copy[date_col] >= period2_start) & 
            (df_copy[date_col] <= period2_end)
        ]
        
        if period1_data.empty or period2_data.empty:
            return {'error': 'One or both periods have no data'}
        
        # Calculate metrics for each period
        def calculate_period_metrics(data):
            return {
                'total': data[value_col].sum(),
                'mean': data[value_col].mean(),
                'median': data[value_col].median(),
                'std': data[value_col].std(),
                'count': len(data),
                'min': data[value_col].min(),
                'max': data[value_col].max()
            }
        
        period1_metrics = calculate_period_metrics(period1_data)
        period2_metrics = calculate_period_metrics(period2_data)
        
        # Calculate changes
        comparison = {
            'period1': {
                'start': period1_start,
                'end': period1_end,
                'metrics': period1_metrics
            },
            'period2': {
                'start': period2_start,
                'end': period2_end,
                'metrics': period2_metrics
            },
            'changes': {}
        }
        
        for metric in ['total', 'mean', 'median', 'count']:
            val1 = period1_metrics[metric]
            val2 = period2_metrics[metric]
            
            if val1 != 0:
                pct_change = ((val2 - val1) / val1) * 100
            else:
                pct_change = float('inf') if val2 > 0 else 0
            
            comparison['changes'][metric] = {
                'absolute_change': val2 - val1,
                'percentage_change': pct_change
            }
        
        return comparison
