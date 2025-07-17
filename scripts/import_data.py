"""Import data script."""

import sys
import logging
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.excel_importer import ExcelImporter
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.database_manager import DatabaseManager
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR


def import_excel_files(file_paths: List[Path] = None, clean_data: bool = True) -> bool:
    """Import Excel files into the database.
    
    Args:
        file_paths: Specific files to import (None for all files in raw directory)
        clean_data: Whether to clean data before importing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize components
        importer = ExcelImporter()
        cleaner = DataCleaner()
        db_manager = DatabaseManager()
        
        # Get files to import
        if file_paths:
            files_to_import = file_paths
        else:
            files_to_import = importer.get_excel_files(RAW_DATA_DIR)
        
        if not files_to_import:
            print("No Excel files found to import.")
            return True
        
        print(f"Found {len(files_to_import)} file(s) to import:")
        for file_path in files_to_import:
            print(f"  - {file_path.name}")
        
        total_imported = 0
        
        for file_path in files_to_import:
            try:
                print(f"\nProcessing: {file_path.name}")
                
                # Read Excel file
                df = importer.read_excel_file(file_path)
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
                
                if df.empty:
                    print(f"  Skipping empty file: {file_path.name}")
                    continue
                
                # Clean data if requested
                if clean_data:
                    print("  Cleaning data...")
                    df_clean, cleaning_report = cleaner.clean_dataframe(df)
                    print(f"  Data cleaning completed: {df.shape} -> {df_clean.shape}")
                    print(f"  Quality score: {cleaning_report['quality_report']['quality_score']:.1f}%")
                    
                    # Save cleaned data
                    cleaned_filename = f"cleaned_{file_path.stem}.csv"
                    cleaner.save_cleaned_data(df_clean, cleaned_filename)
                    print(f"  Saved cleaned data: {cleaned_filename}")
                    
                    df_to_import = df_clean
                else:
                    df_to_import = df
                
                # Import to database
                print("  Importing to database...")
                rows_imported = db_manager.insert_dataframe(df_to_import, 'financial_data', 'append')
                total_imported += rows_imported
                
                print(f"  Successfully imported {rows_imported} rows")
                
                # Record import in data_sources table
                source_record = {
                    'filename': file_path.name,
                    'rows_imported': rows_imported,
                    'status': 'completed'
                }
                db_manager.insert_dataframe(pd.DataFrame([source_record]), 'data_sources', 'append')
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {str(e)}")
                logging.error(f"Failed to import {file_path.name}: {str(e)}")
                continue
        
        print(f"\nImport completed!")
        print(f"Total rows imported: {total_imported:,}")
        
        # Display database statistics
        stats = db_manager.get_database_stats()
        print(f"Database size: {stats['database_size_mb']:.2f} MB")
        
        if 'financial_data' in stats['tables']:
            print(f"Total records in database: {stats['tables']['financial_data']['row_count']:,}")
        
        return True
        
    except Exception as e:
        print(f"Import failed: {str(e)}")
        logging.error(f"Import process failed: {str(e)}")
        return False


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Import Excel files to financial database')
    parser.add_argument('--files', nargs='+', help='Specific files to import')
    parser.add_argument('--no-clean', action='store_true', help='Skip data cleaning')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Financial Analysis Data Import")
    print("=" * 40)
    
    # Prepare file paths
    file_paths = None
    if args.files:
        file_paths = [Path(f) for f in args.files]
        # Validate files exist
        for file_path in file_paths:
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
    
    # Run import
    success = import_excel_files(
        file_paths=file_paths,
        clean_data=not args.no_clean
    )
    
    if success:
        print("\nData import completed successfully!")
        sys.exit(0)
    else:
        print("\nData import failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Import pandas here to avoid issues if not installed
    import pandas as pd
    main()
