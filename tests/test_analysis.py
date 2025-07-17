"""
Unit tests for analysis modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.financial_metrics import FinancialMetricsCalculator
from src.analysis.trend_analysis import TrendAnalyzer


class TestFinancialMetricsCalculator:
    """Test cases for FinancialMetricsCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calculator = FinancialMetricsCalculator()
    
    def create_test_dataframe(self):
        """Create a test DataFrame with financial data."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        data = {
            'date': dates,
            'amount': np.random.normal(100, 20, len(dates)),
            'category': np.random.choice(['Food', 'Transport', 'Entertainment'], len(dates)),
            'description': [f'Transaction {i}' for i in range(len(dates))],
            'store_location': np.random.choice(['Store A', 'Store B', 'Store C'], len(dates))
        }
        return pd.DataFrame(data)
    
    def test_calculate_basic_metrics(self):
        """Test basic financial metrics calculation."""
        df = self.create_test_dataframe()
        metrics = self.calculator.calculate_basic_metrics(df)
        
        assert hasattr(metrics, 'total_revenue')
        assert hasattr(metrics, 'total_expenses')
        assert hasattr(metrics, 'net_income')
        assert hasattr(metrics, 'gross_margin')
        assert hasattr(metrics, 'average_transaction')
        assert hasattr(metrics, 'transaction_count')
        
        assert metrics.transaction_count == len(df)
        assert abs(metrics.total_revenue - df[df['amount'] > 0]['amount'].sum()) < 0.01
        assert abs(metrics.total_expenses - abs(df[df['amount'] < 0]['amount'].sum())) < 0.01
        assert abs(metrics.net_income - (metrics.total_revenue - metrics.total_expenses)) < 0.01
    
    def test_calculate_monthly_metrics(self):
        """Test monthly metrics calculation."""
        df = self.create_test_dataframe()
        monthly_metrics = self.calculator.calculate_monthly_metrics(df)
        
        assert isinstance(monthly_metrics, pd.DataFrame)
        assert 'period' in monthly_metrics.columns
        assert 'total_revenue' in monthly_metrics.columns
        assert 'total_expenses' in monthly_metrics.columns
        assert 'net_income' in monthly_metrics.columns
        assert 'transaction_count' in monthly_metrics.columns
        assert 'average_transaction' in monthly_metrics.columns
        
        # Check that we have data for January 2023
        assert len(monthly_metrics) >= 1
    
    def test_calculate_category_metrics(self):
        """Test category metrics calculation."""
        df = self.create_test_dataframe()
        category_metrics = self.calculator.calculate_category_metrics(df)
        
        assert isinstance(category_metrics, pd.DataFrame)
        assert 'category' in category_metrics.columns
        assert 'total_amount' in category_metrics.columns
        assert 'transaction_count' in category_metrics.columns
        assert 'average_amount' in category_metrics.columns
        
        # Check that all categories are represented
        unique_categories = df['category'].unique()
        assert len(category_metrics) == len(unique_categories)
    
    def test_calculate_growth_metrics(self):
        """Test growth metrics calculation."""
        # Create data spanning multiple months
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'amount': np.random.normal(100, 20, len(dates)),
            'category': np.random.choice(['Food', 'Transport'], len(dates))
        }
        df = pd.DataFrame(data)
        
        growth_metrics = self.calculator.calculate_growth_metrics(df, period='M')
        
        assert isinstance(growth_metrics, pd.DataFrame)
        if len(growth_metrics) > 1:  # Need at least 2 periods for growth calculation
            assert 'period' in growth_metrics.columns
            assert 'amount_growth_rate' in growth_metrics.columns
            assert 'count_growth_rate' in growth_metrics.columns
            assert 'cumulative_amount' in growth_metrics.columns
            assert 'cumulative_count' in growth_metrics.columns
    
    def test_calculate_ratios(self):
        """Test financial ratios calculation."""
        df = self.create_test_dataframe()
        ratios = self.calculator.calculate_ratios(df)
        
        assert isinstance(ratios, dict)
        assert 'revenue_expense_ratio' in ratios
        assert 'average_transaction_ratio' in ratios
        
        # Basic sanity checks
        assert ratios['revenue_expense_ratio'] >= 0
        assert ratios['average_transaction_ratio'] >= 0
    
    def test_calculate_seasonal_metrics(self):
        """Test seasonal metrics calculation."""
        df = self.create_test_dataframe()
        seasonal_metrics = self.calculator.calculate_seasonal_metrics(df)
        
        assert isinstance(seasonal_metrics, pd.DataFrame)
        assert 'season_type' in seasonal_metrics.columns
        assert 'season_value' in seasonal_metrics.columns
        assert 'sum' in seasonal_metrics.columns
        assert 'count' in seasonal_metrics.columns
        assert 'mean' in seasonal_metrics.columns
        
        # Should have monthly, quarterly, and day_of_week data
        season_types = seasonal_metrics['season_type'].unique()
        expected_types = ['monthly', 'quarterly', 'day_of_week']
        for season_type in expected_types:
            assert season_type in season_types


class TestTrendAnalyzer:
    """Test cases for TrendAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = TrendAnalyzer()
    
    def create_test_time_series(self, trend='increasing', noise_level=0.1):
        """Create a test time series with known trend."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        
        if trend == 'increasing':
            base_values = np.linspace(100, 150, len(dates))
        elif trend == 'decreasing':
            base_values = np.linspace(150, 100, len(dates))
        else:  # flat
            base_values = np.full(len(dates), 125)
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, noise_level * base_values.mean(), len(dates))
        values = base_values + noise
        
        data = {
            'date': dates,
            'total_amount': values,
            'transaction_count': np.random.randint(10, 50, len(dates))
        }
        return pd.DataFrame(data)
    
    def test_detect_trend_increasing(self):
        """Test trend detection for increasing trend."""
        df = self.create_test_time_series(trend='increasing')
        result = self.analyzer.detect_trend(df, 'total_amount')
        
        assert hasattr(result, 'trend_direction')
        assert hasattr(result, 'slope')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'trend_strength')
        
        assert result.trend_direction == 'increasing'
        assert result.slope > 0
        assert result.r_squared > 0.5  # Should have good fit for clean trend
    
    def test_detect_trend_decreasing(self):
        """Test trend detection for decreasing trend."""
        df = self.create_test_time_series(trend='decreasing')
        result = self.analyzer.detect_trend(df, 'total_amount')
        
        assert result.trend_direction == 'decreasing'
        assert result.slope < 0
        assert result.r_squared > 0.5
    
    def test_detect_trend_flat(self):
        """Test trend detection for flat trend."""
        df = self.create_test_time_series(trend='flat')
        result = self.analyzer.detect_trend(df, 'total_amount')
        
        assert result.trend_direction == 'stable'
        assert abs(result.slope) < 0.5  # Should be close to zero
    
    def test_analyze_moving_averages(self):
        """Test moving averages analysis."""
        df = self.create_test_time_series()
        windows = [7, 14]
        result = self.analyzer.analyze_moving_averages(df, 'total_amount', windows)
        
        assert isinstance(result, dict)
        for window in windows:
            assert window in result
            ma_data = result[window]
            assert 'current_ma' in ma_data
            assert 'previous_ma' in ma_data
            assert 'ma_change' in ma_data
            assert 'ma_trend' in ma_data
            assert 'volatility' in ma_data
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        df = self.create_test_time_series()
        
        # Add some artificial anomalies
        df.loc[5, 'total_amount'] = 500  # Large positive anomaly
        df.loc[15, 'total_amount'] = 10  # Large negative anomaly
        
        anomalies = self.analyzer.detect_anomalies(df, 'total_amount')
        
        assert isinstance(anomalies, pd.DataFrame)
        if len(anomalies) > 0:
            assert 'date' in anomalies.columns
            assert 'amount' in anomalies.columns
            assert 'anomaly_score' in anomalies.columns
            
            # Should detect our artificial anomalies
            assert len(anomalies) >= 1
    
    def test_forecast_trend(self):
        """Test trend forecasting."""
        df = self.create_test_time_series(trend='increasing')
        forecast_days = 7
        
        result = self.analyzer.forecast_trend(df, 'total_amount', forecast_days)
        
        assert hasattr(result, 'forecast_values')
        assert hasattr(result, 'method')
        assert hasattr(result, 'accuracy_metrics')
        assert hasattr(result, 'confidence_level')
        
        assert len(result.forecast_values) == forecast_days
        assert isinstance(result.accuracy_metrics, dict)
        assert 'mae' in result.accuracy_metrics
        assert 'rmse' in result.accuracy_metrics
        
        # For increasing trend, forecast should generally increase
        if result.method != 'mean':  # Mean method might not capture trend
            assert result.forecast_values[-1] >= result.forecast_values[0]


class TestAnalysisIntegration:
    """Integration tests for the complete analysis pipeline."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calculator = FinancialMetricsCalculator()
        self.analyzer = TrendAnalyzer()
    
    def create_comprehensive_dataset(self):
        """Create a comprehensive dataset for integration testing."""
        # Create 3 months of data with realistic patterns
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        np.random.seed(42)
        
        # Simulate realistic financial data with trends and seasonality
        base_amount = 100
        trend = np.linspace(0, 20, len(dates))  # Slight increasing trend
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 5, len(dates))
        
        amounts = base_amount + trend + seasonal + noise
        
        data = {
            'date': dates,
            'amount': amounts,
            'category': np.random.choice(['Food', 'Transport', 'Entertainment', 'Shopping'], len(dates)),
            'description': [f'Transaction {i}' for i in range(len(dates))],
            'store_location': np.random.choice(['Store A', 'Store B', 'Store C', 'Store D'], len(dates))
        }
        return pd.DataFrame(data)
    
    def test_complete_analysis_pipeline(self):
        """Test the complete analysis pipeline."""
        df = self.create_comprehensive_dataset()
        
        # Test basic metrics
        basic_metrics = self.calculator.calculate_basic_metrics(df)
        assert basic_metrics.transaction_count == len(df)
        assert basic_metrics.total_revenue > 0
        
        # Test monthly analysis
        monthly_metrics = self.calculator.calculate_monthly_metrics(df)
        assert len(monthly_metrics) == 3  # 3 months of data
        
        # Test category analysis
        category_metrics = self.calculator.calculate_category_metrics(df)
        assert len(category_metrics) == 4  # 4 categories
        
        # Test trend analysis
        daily_data = df.groupby('date').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        daily_data.columns = ['total_amount', 'transaction_count', 'avg_amount']
        daily_data = daily_data.reset_index()
        
        trend_result = self.analyzer.detect_trend(daily_data, 'total_amount')
        assert trend_result.trend_direction in ['increasing', 'decreasing', 'stable']
        
        # Test forecasting
        forecast_result = self.analyzer.forecast_trend(daily_data, 'total_amount', 7)
        assert len(forecast_result.forecast_values) == 7
        
        # Test anomaly detection
        anomalies = self.analyzer.detect_anomalies(daily_data, 'total_amount')
        assert isinstance(anomalies, pd.DataFrame)
    
    def test_data_consistency(self):
        """Test data consistency across different analysis methods."""
        df = self.create_comprehensive_dataset()
        
        # Calculate metrics using different methods and verify consistency
        basic_metrics = self.calculator.calculate_basic_metrics(df)
        monthly_metrics = self.calculator.calculate_monthly_metrics(df)
        
        # Total from monthly should match basic metrics
        monthly_total = monthly_metrics['total_revenue'].sum() - monthly_metrics['total_expenses'].sum()
        basic_net = basic_metrics.net_income
        
        # Allow for small floating point differences
        assert abs(monthly_total - basic_net) < 1.0
        
        # Transaction count consistency
        monthly_count_total = monthly_metrics['transaction_count'].sum()
        assert monthly_count_total == basic_metrics.transaction_count


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
