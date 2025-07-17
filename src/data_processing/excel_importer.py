"""Excel file import functionality."""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.settings import RAW_DATA_DIR, SUPPORTED_FILE_TYPES, MAX_FILE_SIZE_MB


class ExcelImporter:
    """Handle Excel file imports and basic data extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_excel_files(self, directory: Path = RAW_DATA_DIR) -> List[Path]:
        """Get list of Excel files in the specified directory.
        
        Args:
            directory: Directory to search for Excel files
            
        Returns:
            List of Excel file paths
        """
        excel_files = []
        for file_type in SUPPORTED_FILE_TYPES:
            excel_files.extend(directory.glob(f"*{file_type}"))
        
        return excel_files
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file can be processed.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        if not file_path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return False
            
        if file_path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            self.logger.error(f"Unsupported file type: {file_path.suffix}")
            return False
            
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            self.logger.error(f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB")
            return False
            
        return True
    
    def read_excel_file(self, file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Read Excel file and return DataFrame.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet to read (None for first sheet)
            
        Returns:
            DataFrame with the Excel data
        """
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
            
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
            self.logger.info(f"Successfully read {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def get_sheet_names(self, file_path: Path) -> List[str]:
        """Get list of sheet names from Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of sheet names
        """
        if not self.validate_file(file_path):
            return []
            
        try:
            if file_path.suffix.lower() == '.csv':
                return ['Sheet1']  # CSV files don't have sheet names
            else:
                excel_file = pd.ExcelFile(file_path)
                return excel_file.sheet_names
                
        except Exception as e:
            self.logger.error(f"Error getting sheet names from {file_path}: {str(e)}")
            return []
    
    def import_all_files(self, directory: Path = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
        """Import all Excel files from directory.
        
        Args:
            directory: Directory containing Excel files
            
        Returns:
            Dictionary mapping file names to DataFrames
        """
        files = self.get_excel_files(directory)
        imported_data = {}
        
        for file_path in files:
            try:
                df = self.read_excel_file(file_path)
                imported_data[file_path.stem] = df
                self.logger.info(f"Imported {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to import {file_path.name}: {str(e)}")
                
        return imported_data
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about an Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary with file information
        """
        if not self.validate_file(file_path):
            return {}
            
        info = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_type': file_path.suffix,
            'sheet_names': self.get_sheet_names(file_path)
        }
        
        try:
            df = self.read_excel_file(file_path)
            info.update({
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            })
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
            
        return info
