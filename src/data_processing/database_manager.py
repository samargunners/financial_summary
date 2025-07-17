"""Database management functionality."""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from config.settings import DATABASE_PATH, DATABASE_DIR


class DatabaseManager:
    """Handle SQLite database operations."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection.
        
        Returns:
            SQLite connection object
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def create_tables(self) -> None:
        """Create necessary database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Financial data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS financial_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT,
                    subcategory TEXT,
                    description TEXT,
                    store_location TEXT,
                    payment_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metadata table for tracking data sources
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT,
                    import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rows_imported INTEGER,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Summary statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS summary_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    calculation_date DATE,
                    category TEXT,
                    period TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_financial_data_date 
                ON financial_data (date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_financial_data_category 
                ON financial_data (category)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_financial_data_amount 
                ON financial_data (amount)
            ''')
            
            conn.commit()
            self.logger.info("Database tables created successfully")
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                        if_exists: str = 'append') -> int:
        """Insert DataFrame into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows inserted
        """
        try:
            with self.get_connection() as conn:
                rows_inserted = df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                self.logger.info(f"Inserted {len(df)} rows into {table_name}")
                return len(df)
                
        except Exception as e:
            self.logger.error(f"Error inserting data into {table_name}: {str(e)}")
            raise
    
    def query_data(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            Query results as DataFrame
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                self.logger.info(f"Query returned {len(df)} rows")
                return df
                
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            return {
                'name': table_name,
                'columns': [dict(col) for col in columns],
                'row_count': row_count
            }
    
    def list_tables(self) -> List[str]:
        """Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            return tables
    
    def delete_data(self, table_name: str, where_clause: str, params: Optional[tuple] = None) -> int:
        """Delete data from table.
        
        Args:
            table_name: Target table name
            where_clause: WHERE clause for deletion
            params: Query parameters (optional)
            
        Returns:
            Number of rows deleted
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = f"DELETE FROM {table_name} WHERE {where_clause}"
                cursor.execute(query, params or ())
                rows_deleted = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Deleted {rows_deleted} rows from {table_name}")
                return rows_deleted
                
        except Exception as e:
            self.logger.error(f"Error deleting data from {table_name}: {str(e)}")
            raise
    
    def update_data(self, table_name: str, set_clause: str, where_clause: str, 
                   params: Optional[tuple] = None) -> int:
        """Update data in table.
        
        Args:
            table_name: Target table name
            set_clause: SET clause for update
            where_clause: WHERE clause for update
            params: Query parameters (optional)
            
        Returns:
            Number of rows updated
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
                cursor.execute(query, params or ())
                rows_updated = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Updated {rows_updated} rows in {table_name}")
                return rows_updated
                
        except Exception as e:
            self.logger.error(f"Error updating data in {table_name}: {str(e)}")
            raise
    
    def backup_database(self, backup_path: Path) -> None:
        """Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file
        """
        try:
            with self.get_connection() as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
                    
            self.logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error backing up database: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            'database_path': str(self.db_path),
            'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
            'tables': {}
        }
        
        tables = self.list_tables()
        for table in tables:
            try:
                info = self.get_table_info(table)
                stats['tables'][table] = {
                    'row_count': info['row_count'],
                    'column_count': len(info['columns'])
                }
            except Exception as e:
                self.logger.warning(f"Could not get stats for table {table}: {str(e)}")
                
        return stats
    
    def execute_script(self, script_path: Path) -> None:
        """Execute SQL script from file.
        
        Args:
            script_path: Path to SQL script file
        """
        try:
            with open(script_path, 'r') as f:
                script = f.read()
                
            with self.get_connection() as conn:
                conn.executescript(script)
                
            self.logger.info(f"Executed SQL script: {script_path}")
            
        except Exception as e:
            self.logger.error(f"Error executing script {script_path}: {str(e)}")
            raise
