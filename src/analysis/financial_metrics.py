"""Financial metrics calculation functionality."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class FinancialMetrics:
    """Data class for storing financial metrics."""
    total_revenue: float
    total_expenses: float
    net_income: float
    gross_margin: float
    average_transaction: float
    transaction_count: int
    period_start: datetime
    period_end: datetime


class FinancialMetricsCalculator:
    """Calculate various financial metrics from transaction data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(self, df: pd.DataFrame, 
                              amount_col: str = 'amount',
                              date_col: str = 'date') -> FinancialMetrics:
        """Calculate basic financial metrics.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            date_col: Name of date column
            
        Returns:
            FinancialMetrics object with calculated values
        """
        if df.empty:
            return FinancialMetrics(0, 0, 0, 0, 0, 0, datetime.now(), datetime.now())
        
        # Separate revenue (positive) and expenses (negative)
        revenue_data = df[df[amount_col] > 0]
        expense_data = df[df[amount_col] < 0]
        
        total_revenue = revenue_data[amount_col].sum()
        total_expenses = abs(expense_data[amount_col].sum())
        net_income = total_revenue - total_expenses
        
        # Calculate gross margin
        gross_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0
        
        # Transaction metrics
        average_transaction = df[amount_col].mean()
        transaction_count = len(df)
        
        # Date range
        period_start = pd.to_datetime(df[date_col]).min()
        period_end = pd.to_datetime(df[date_col]).max()
        
        return FinancialMetrics(
            total_revenue=total_revenue,
            total_expenses=total_expenses,
            net_income=net_income,
            gross_margin=gross_margin,
            average_transaction=average_transaction,
            transaction_count=transaction_count,
            period_start=period_start,
            period_end=period_end
        )
    
    def calculate_monthly_metrics(self, df: pd.DataFrame,
                                 amount_col: str = 'amount',
                                 date_col: str = 'date') -> pd.DataFrame:
        """Calculate monthly financial metrics.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            date_col: Name of date column
            
        Returns:
            DataFrame with monthly metrics
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy['year_month'] = df_copy[date_col].dt.to_period('M')
        
        monthly_metrics = []
        
        for period in df_copy['year_month'].unique():
            period_data = df_copy[df_copy['year_month'] == period]
            metrics = self.calculate_basic_metrics(period_data, amount_col, date_col)
            
            monthly_metrics.append({
                'period': str(period),
                'year': period.year,
                'month': period.month,
                'total_revenue': metrics.total_revenue,
                'total_expenses': metrics.total_expenses,
                'net_income': metrics.net_income,
                'gross_margin': metrics.gross_margin,
                'average_transaction': metrics.average_transaction,
                'transaction_count': metrics.transaction_count
            })
        
        return pd.DataFrame(monthly_metrics).sort_values('period')
    
    def calculate_category_metrics(self, df: pd.DataFrame,
                                  amount_col: str = 'amount',
                                  category_col: str = 'category') -> pd.DataFrame:
        """Calculate metrics by category.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            category_col: Name of category column
            
        Returns:
            DataFrame with category metrics
        """
        if category_col not in df.columns:
            self.logger.warning(f"Category column '{category_col}' not found")
            return pd.DataFrame()
        
        category_metrics = []
        
        for category in df[category_col].unique():
            if pd.isna(category):
                continue
                
            category_data = df[df[category_col] == category]
            
            total_amount = category_data[amount_col].sum()
            transaction_count = len(category_data)
            average_amount = category_data[amount_col].mean()
            
            # Calculate percentage of total
            total_all = df[amount_col].sum()
            percentage_of_total = (total_amount / total_all * 100) if total_all != 0 else 0
            
            category_metrics.append({
                'category': category,
                'total_amount': total_amount,
                'transaction_count': transaction_count,
                'average_amount': average_amount,
                'percentage_of_total': percentage_of_total,
                'min_amount': category_data[amount_col].min(),
                'max_amount': category_data[amount_col].max()
            })
        
        return pd.DataFrame(category_metrics).sort_values('total_amount', ascending=False)
    
    def calculate_growth_metrics(self, df: pd.DataFrame,
                               amount_col: str = 'amount',
                               date_col: str = 'date',
                               period: str = 'M') -> pd.DataFrame:
        """Calculate growth metrics over time.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            date_col: Name of date column
            period: Time period for grouping ('M' for monthly, 'Q' for quarterly)
            
        Returns:
            DataFrame with growth metrics
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Group by period
        grouped = df_copy.groupby(df_copy[date_col].dt.to_period(period))[amount_col].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        
        grouped.columns = ['period', 'total_amount', 'transaction_count', 'average_amount']
        
        # Calculate growth rates
        grouped['amount_growth_rate'] = grouped['total_amount'].pct_change() * 100
        grouped['count_growth_rate'] = grouped['transaction_count'].pct_change() * 100
        grouped['average_growth_rate'] = grouped['average_amount'].pct_change() * 100
        
        # Calculate cumulative values
        grouped['cumulative_amount'] = grouped['total_amount'].cumsum()
        grouped['cumulative_count'] = grouped['transaction_count'].cumsum()
        
        return grouped
    
    def calculate_ratios(self, df: pd.DataFrame,
                        amount_col: str = 'amount') -> Dict[str, float]:
        """Calculate financial ratios.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            
        Returns:
            Dictionary with calculated ratios
        """
        if df.empty:
            return {}
        
        amounts = df[amount_col]
        positive_amounts = amounts[amounts > 0]
        negative_amounts = amounts[amounts < 0]
        
        ratios = {
            'revenue_to_expense_ratio': (positive_amounts.sum() / abs(negative_amounts.sum())) if negative_amounts.sum() != 0 else float('inf'),
            'average_positive_transaction': positive_amounts.mean() if len(positive_amounts) > 0 else 0,
            'average_negative_transaction': negative_amounts.mean() if len(negative_amounts) > 0 else 0,
            'transaction_value_variance': amounts.var(),
            'transaction_value_std': amounts.std(),
            'coefficient_of_variation': (amounts.std() / amounts.mean()) if amounts.mean() != 0 else 0
        }
        
        return ratios
    
    def calculate_seasonal_metrics(self, df: pd.DataFrame,
                                 amount_col: str = 'amount',
                                 date_col: str = 'date') -> pd.DataFrame:
        """Calculate seasonal patterns in financial data.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            date_col: Name of date column
            
        Returns:
            DataFrame with seasonal metrics
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Extract seasonal components
        df_copy['month'] = df_copy[date_col].dt.month
        df_copy['quarter'] = df_copy[date_col].dt.quarter
        df_copy['day_of_week'] = df_copy[date_col].dt.day_name()
        df_copy['week_of_year'] = df_copy[date_col].dt.isocalendar().week
        
        seasonal_metrics = []
        
        # Monthly seasonality
        monthly = df_copy.groupby('month')[amount_col].agg(['sum', 'count', 'mean']).reset_index()
        monthly['season_type'] = 'monthly'
        monthly['season_value'] = monthly['month']
        seasonal_metrics.append(monthly[['season_type', 'season_value', 'sum', 'count', 'mean']])
        
        # Quarterly seasonality
        quarterly = df_copy.groupby('quarter')[amount_col].agg(['sum', 'count', 'mean']).reset_index()
        quarterly['season_type'] = 'quarterly'
        quarterly['season_value'] = quarterly['quarter']
        seasonal_metrics.append(quarterly[['season_type', 'season_value', 'sum', 'count', 'mean']])
        
        # Day of week seasonality
        dow = df_copy.groupby('day_of_week')[amount_col].agg(['sum', 'count', 'mean']).reset_index()
        dow['season_type'] = 'day_of_week'
        dow['season_value'] = dow['day_of_week']
        seasonal_metrics.append(dow[['season_type', 'season_value', 'sum', 'count', 'mean']])
        
        return pd.concat(seasonal_metrics, ignore_index=True)
    
    def calculate_benchmark_metrics(self, df: pd.DataFrame,
                                  amount_col: str = 'amount',
                                  benchmarks: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate performance against benchmarks.
        
        Args:
            df: DataFrame with financial data
            amount_col: Name of amount column
            benchmarks: Dictionary of benchmark values
            
        Returns:
            Dictionary with benchmark comparison results
        """
        if benchmarks is None:
            benchmarks = {
                'target_monthly_revenue': 10000,
                'target_transaction_count': 100,
                'target_average_transaction': 100
            }
        
        metrics = self.calculate_basic_metrics(df, amount_col)
        
        # Calculate monthly averages for comparison
        monthly_metrics = self.calculate_monthly_metrics(df, amount_col)
        avg_monthly_revenue = monthly_metrics['total_revenue'].mean() if not monthly_metrics.empty else 0
        avg_monthly_count = monthly_metrics['transaction_count'].mean() if not monthly_metrics.empty else 0
        
        benchmark_results = {
            'actual_vs_target': {
                'monthly_revenue': {
                    'actual': avg_monthly_revenue,
                    'target': benchmarks.get('target_monthly_revenue', 0),
                    'variance': avg_monthly_revenue - benchmarks.get('target_monthly_revenue', 0),
                    'achievement_rate': (avg_monthly_revenue / benchmarks.get('target_monthly_revenue', 1)) * 100
                },
                'transaction_count': {
                    'actual': avg_monthly_count,
                    'target': benchmarks.get('target_transaction_count', 0),
                    'variance': avg_monthly_count - benchmarks.get('target_transaction_count', 0),
                    'achievement_rate': (avg_monthly_count / benchmarks.get('target_transaction_count', 1)) * 100
                },
                'average_transaction': {
                    'actual': metrics.average_transaction,
                    'target': benchmarks.get('target_average_transaction', 0),
                    'variance': metrics.average_transaction - benchmarks.get('target_average_transaction', 0),
                    'achievement_rate': (metrics.average_transaction / benchmarks.get('target_average_transaction', 1)) * 100
                }
            }
        }
        
        return benchmark_results
