"""
Unit tests for data processing modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import sqlite3

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.excel_importer import ExcelImporter
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.database_manager import DatabaseManager


class TestExcelImporter:
    """Test cases for ExcelImporter class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.importer = ExcelImporter()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_excel_files_empty_directory(self):
        """Test getting Excel files from empty directory."""
        files = self.importer.get_excel_files(self.temp_dir)
        assert files == []
    
    def test_get_excel_files_with_excel_files(self):
        """Test getting Excel files from directory with Excel files."""
        # Create test Excel files
        test_files = ['test1.xlsx', 'test2.xls', 'test3.csv', 'not_excel.txt']
        for file in test_files:
            Path(self.temp_dir, file).touch()
        
        files = self.importer.get_excel_files(self.temp_dir)
        excel_files = [f for f in files if f.endswith(('.xlsx', '.xls', '.csv'))]
        
        assert len(excel_files) == 3
        assert any('test1.xlsx' in f for f in excel_files)
        assert any('test2.xls' in f for f in excel_files)
        assert any('test3.csv' in f for f in excel_files)
    
    def test_validate_file_valid_path(self):
        """Test file validation with valid file path."""
        test_file = Path(self.temp_dir, 'test.xlsx')
        test_file.touch()
        
        result = self.importer.validate_file(str(test_file))
        assert result['is_valid'] is True
        assert result['file_size'] >= 0
    
    def test_validate_file_invalid_path(self):
        """Test file validation with invalid file path."""
        result = self.importer.validate_file('nonexistent_file.xlsx')
        assert result['is_valid'] is False
        assert 'error' in result
    
    def test_read_excel_file_nonexistent(self):
        """Test reading non-existent Excel file."""
        result = self.importer.read_excel_file('nonexistent_file.xlsx')
        assert result is None


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.cleaner = DataCleaner()
    
    def create_test_dataframe(self):
        """Create a test DataFrame with various data quality issues."""
        data = {
            'Date ': ['2023-01-01', '2023-01-02', '', '2023-01-04'],
            ' Amount  ': [100.50, -50.25, 0, 200.75],
            'Category': ['Food', 'Transport', None, 'Entertainment'],
            'Description': ['Lunch', '', 'Missing', 'Movie tickets'],
            'Store Location': ['Store A', 'Store B', 'Store A', '']
        }
        return pd.DataFrame(data)
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        df = self.create_test_dataframe()
        cleaned_df = self.cleaner.clean_column_names(df)
        
        expected_columns = ['date', 'amount', 'category', 'description', 'store_location']
        assert list(cleaned_df.columns) == expected_columns
    
    def test_clean_dates(self):
        """Test date cleaning and conversion."""
        df = self.create_test_dataframe()
        df = self.cleaner.clean_column_names(df)
        cleaned_df = self.cleaner.clean_dates(df, 'date')
        
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['date'])
        assert not cleaned_df['date'].isna().all()
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = self.create_test_dataframe()
        df = self.cleaner.clean_column_names(df)
        cleaned_df = self.cleaner.handle_missing_values(df)
        
        # Check that missing values are handled appropriately
        assert not cleaned_df['category'].isna().any()
        assert not cleaned_df['description'].isna().any()
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        df = self.create_test_dataframe()
        df = self.cleaner.clean_column_names(df)
        
        report = self.cleaner.validate_data_quality(df)
        
        assert 'missing_values' in report
        assert 'data_types' in report
        assert 'row_count' in report
        assert report['row_count'] == len(df)
    
    def test_clean_dataframe_complete_pipeline(self):
        """Test complete data cleaning pipeline."""
        df = self.create_test_dataframe()
        cleaned_df = self.cleaner.clean_dataframe(df)
        
        # Verify the cleaning pipeline worked
        assert list(cleaned_df.columns) == ['date', 'amount', 'category', 'description', 'store_location']
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['date'])
        assert pd.api.types.is_numeric_dtype(cleaned_df['amount'])


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up after each test method."""
        self.db_manager.close_connection()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_create_tables(self):
        """Test table creation."""
        self.db_manager.create_tables()
        
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'financial_data' in tables
        conn.close()
    
    def test_insert_dataframe(self):
        """Test DataFrame insertion."""
        self.db_manager.create_tables()
        
        # Create test DataFrame
        data = {
            'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'amount': [100.50, -50.25],
            'category': ['Food', 'Transport'],
            'description': ['Lunch', 'Bus fare'],
            'store_location': ['Store A', 'Store B']
        }
        df = pd.DataFrame(data)
        
        result = self.db_manager.insert_dataframe(df, 'financial_data')
        assert result is True
        
        # Verify data was inserted
        query_result = self.db_manager.query_data("SELECT COUNT(*) as count FROM financial_data")
        assert len(query_result) == 1
        assert query_result.iloc[0]['count'] == 2
    
    def test_query_data(self):
        """Test data querying."""
        self.db_manager.create_tables()
        
        # Insert test data
        data = {
            'date': [datetime(2023, 1, 1)],
            'amount': [100.50],
            'category': ['Food'],
            'description': ['Lunch'],
            'store_location': ['Store A']
        }
        df = pd.DataFrame(data)
        self.db_manager.insert_dataframe(df, 'financial_data')
        
        # Query data
        result = self.db_manager.query_data("SELECT * FROM financial_data")
        
        assert len(result) == 1
        assert result.iloc[0]['amount'] == 100.50
        assert result.iloc[0]['category'] == 'Food'
    
    def test_update_record(self):
        """Test record updating."""
        self.db_manager.create_tables()
        
        # Insert test data
        data = {
            'date': [datetime(2023, 1, 1)],
            'amount': [100.50],
            'category': ['Food'],
            'description': ['Lunch'],
            'store_location': ['Store A']
        }
        df = pd.DataFrame(data)
        self.db_manager.insert_dataframe(df, 'financial_data')
        
        # Update record
        update_query = "UPDATE financial_data SET amount = 150.75 WHERE category = 'Food'"
        result = self.db_manager.execute_query(update_query)
        assert result is True
        
        # Verify update
        query_result = self.db_manager.query_data("SELECT amount FROM financial_data WHERE category = 'Food'")
        assert query_result.iloc[0]['amount'] == 150.75
    
    def test_delete_record(self):
        """Test record deletion."""
        self.db_manager.create_tables()
        
        # Insert test data
        data = {
            'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'amount': [100.50, -50.25],
            'category': ['Food', 'Transport'],
            'description': ['Lunch', 'Bus fare'],
            'store_location': ['Store A', 'Store B']
        }
        df = pd.DataFrame(data)
        self.db_manager.insert_dataframe(df, 'financial_data')
        
        # Delete record
        delete_query = "DELETE FROM financial_data WHERE category = 'Food'"
        result = self.db_manager.execute_query(delete_query)
        assert result is True
        
        # Verify deletion
        query_result = self.db_manager.query_data("SELECT COUNT(*) as count FROM financial_data")
        assert query_result.iloc[0]['count'] == 1
    
    def test_backup_database(self):
        """Test database backup functionality."""
        self.db_manager.create_tables()
        
        # Insert test data
        data = {
            'date': [datetime(2023, 1, 1)],
            'amount': [100.50],
            'category': ['Food'],
            'description': ['Lunch'],
            'store_location': ['Store A']
        }
        df = pd.DataFrame(data)
        self.db_manager.insert_dataframe(df, 'financial_data')
        
        # Create backup
        backup_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        backup_file.close()
        
        try:
            result = self.db_manager.backup_database(backup_file.name)
            assert result is True
            assert os.path.exists(backup_file.name)
            assert os.path.getsize(backup_file.name) > 0
        finally:
            if os.path.exists(backup_file.name):
                os.unlink(backup_file.name)


# Integration tests
class TestDataProcessingIntegration:
    """Integration tests for the complete data processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.importer = ExcelImporter()
        self.cleaner = DataCleaner()
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.db_manager.close_connection()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_complete_pipeline(self):
        """Test the complete data processing pipeline."""
        # Create test CSV file
        test_data = {
            'Date ': ['2023-01-01', '2023-01-02', '2023-01-03'],
            ' Amount  ': [100.50, -50.25, 200.75],
            'Category': ['Food', 'Transport', 'Entertainment'],
            'Description': ['Lunch', 'Bus fare', 'Movie tickets'],
            'Store Location': ['Store A', 'Store B', 'Store C']
        }
        test_df = pd.DataFrame(test_data)
        test_file = Path(self.temp_dir, 'test_data.csv')
        test_df.to_csv(test_file, index=False)
        
        # Step 1: Import data
        imported_df = self.importer.read_excel_file(str(test_file))
        assert imported_df is not None
        assert len(imported_df) == 3
        
        # Step 2: Clean data
        cleaned_df = self.cleaner.clean_dataframe(imported_df)
        assert list(cleaned_df.columns) == ['date', 'amount', 'category', 'description', 'store_location']
        
        # Step 3: Store in database
        self.db_manager.create_tables()
        result = self.db_manager.insert_dataframe(cleaned_df, 'financial_data')
        assert result is True
        
        # Step 4: Verify data integrity
        query_result = self.db_manager.query_data("SELECT * FROM financial_data ORDER BY date")
        assert len(query_result) == 3
        assert query_result.iloc[0]['category'] == 'Food'
        assert query_result.iloc[1]['category'] == 'Transport'
        assert query_result.iloc[2]['category'] == 'Entertainment'


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
