"""Data cleaning and validation functionality."""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from config.settings import (
    DEFAULT_DATE_FORMAT, MIN_DATE_YEAR, MAX_DATE_YEAR, 
    REQUIRED_COLUMNS, PROCESSED_DATA_DIR
)


class DataCleaner:
    """Handle data cleaning and validation operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        df_clean = df.copy()
        
        # Convert to lowercase and replace spaces/special chars with underscores
        df_clean.columns = (df_clean.columns
                           .str.lower()
                           .str.replace(' ', '_')
                           .str.replace('[^a-zA-Z0-9_]', '', regex=True))
        
        self.logger.info(f"Cleaned column names: {list(df_clean.columns)}")
        return df_clean
    
    def detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that likely contain dates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that appear to contain dates
        """
        date_columns = []
        
        for col in df.columns:
            # Check column name for date-related keywords
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated']):
                date_columns.append(col)
                continue
                
            # Check data types and sample values
            sample_values = df[col].dropna().head(10)
            if len(sample_values) == 0:
                continue
                
            # Try to parse as datetime
            try:
                pd.to_datetime(sample_values, errors='raise')
                date_columns.append(col)
            except:
                continue
                
        self.logger.info(f"Detected date columns: {date_columns}")
        return date_columns
    
    def clean_dates(self, df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Clean and standardize date columns.
        
        Args:
            df: Input DataFrame
            date_columns: Specific columns to treat as dates (auto-detect if None)
            
        Returns:
            DataFrame with cleaned date columns
        """
        df_clean = df.copy()
        
        if date_columns is None:
            date_columns = self.detect_date_columns(df_clean)
            
        for col in date_columns:
            if col not in df_clean.columns:
                continue
                
            try:
                # Convert to datetime
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                
                # Filter out invalid years
                mask = (df_clean[col].dt.year >= MIN_DATE_YEAR) & (df_clean[col].dt.year <= MAX_DATE_YEAR)
                df_clean.loc[~mask, col] = pd.NaT
                
                self.logger.info(f"Cleaned date column: {col}")
                
            except Exception as e:
                self.logger.warning(f"Could not clean date column {col}: {str(e)}")
                
        return df_clean
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns by removing non-numeric characters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned numeric columns
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to numeric after cleaning
                try:
                    # Remove common non-numeric characters
                    cleaned_series = (df_clean[col]
                                    .astype(str)
                                    .str.replace('$', '', regex=False)
                                    .str.replace(',', '', regex=False)
                                    .str.replace('(', '-', regex=False)
                                    .str.replace(')', '', regex=False)
                                    .str.strip())
                    
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If more than 50% of values are numeric, replace the column
                    if numeric_series.notna().sum() / len(numeric_series) > 0.5:
                        df_clean[col] = numeric_series
                        self.logger.info(f"Converted column to numeric: {col}")
                        
                except Exception as e:
                    self.logger.warning(f"Could not clean numeric column {col}: {str(e)}")
                    
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicate detection
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset)
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} duplicate rows")
            
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'fill_mean', 'fill_median', 'fill_zero'
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'fill_mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'fill_median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'fill_zero':
            df_clean = df_clean.fillna(0)
            
        self.logger.info(f"Handled missing values using strategy: {strategy}")
        return df_clean
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
        }
        
        # Check for required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        report['missing_required_columns'] = missing_required
        
        # Data quality score (0-100)
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_penalty = df.duplicated().sum() * len(df.columns)
        
        if total_cells > 0:
            quality_score = max(0, 100 - ((missing_cells + duplicate_penalty) / total_cells * 100))
        else:
            quality_score = 0
            
        report['quality_score'] = round(quality_score, 2)
        
        return report
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       clean_columns: bool = True,
                       clean_dates: bool = True,
                       clean_numeric: bool = True,
                       remove_duplicates: bool = True,
                       handle_missing: str = 'drop') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive data cleaning pipeline.
        
        Args:
            df: Input DataFrame
            clean_columns: Whether to clean column names
            clean_dates: Whether to clean date columns
            clean_numeric: Whether to clean numeric columns
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: Strategy for handling missing values
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        df_clean = df.copy()
        report = {'original_shape': df.shape}
        
        # Clean column names
        if clean_columns:
            df_clean = self.clean_column_names(df_clean)
            
        # Clean dates
        if clean_dates:
            df_clean = self.clean_dates(df_clean)
            
        # Clean numeric columns
        if clean_numeric:
            df_clean = self.clean_numeric_columns(df_clean)
            
        # Remove duplicates
        if remove_duplicates:
            df_clean = self.remove_duplicates(df_clean)
            
        # Handle missing values
        if handle_missing:
            df_clean = self.handle_missing_values(df_clean, handle_missing)
            
        # Generate final report
        report.update({
            'final_shape': df_clean.shape,
            'rows_removed': df.shape[0] - df_clean.shape[0],
            'quality_report': self.validate_data_quality(df_clean)
        })
        
        self.logger.info(f"Data cleaning completed. Shape: {df.shape} -> {df_clean.shape}")
        return df_clean, report
    
    def save_cleaned_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save cleaned data to processed directory.
        
        Args:
            df: Cleaned DataFrame
            filename: Output filename
        """
        output_path = PROCESSED_DATA_DIR / filename
        
        if filename.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif filename.endswith('.json'):
            df.to_json(output_path, orient='records', date_format='iso')
        else:
            # Default to CSV
            output_path = output_path.with_suffix('.csv')
            df.to_csv(output_path, index=False)
            
        self.logger.info(f"Saved cleaned data to {output_path}")
