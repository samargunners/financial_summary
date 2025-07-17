"""Setup database script."""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.database_manager import DatabaseManager
from config.settings import DATABASE_PATH, DATABASE_DIR


def setup_database():
    """Initialize the database with required tables."""
    try:
        # Ensure database directory exists
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        print(f"Setting up database at: {DATABASE_PATH}")
        
        # Create tables
        db_manager.create_tables()
        
        print("Database setup completed successfully!")
        print(f"Database location: {DATABASE_PATH}")
        
        # Display database information
        stats = db_manager.get_database_stats()
        print(f"Database size: {stats['database_size_mb']:.2f} MB")
        print(f"Tables created: {list(stats['tables'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        logging.error(f"Database setup failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Financial Analysis Database Setup")
    print("=" * 40)
    
    success = setup_database()
    
    if success:
        print("\nDatabase setup completed successfully!")
        sys.exit(0)
    else:
        print("\nDatabase setup failed!")
        sys.exit(1)
