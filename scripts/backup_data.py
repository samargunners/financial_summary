"""Backup data script."""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.database_manager import DatabaseManager
from config.settings import DATABASE_PATH, DATA_DIR


def backup_database(backup_dir: Path = None, include_timestamp: bool = True) -> bool:
    """Create a backup of the database.
    
    Args:
        backup_dir: Directory to store backup (default: data/backups)
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Set default backup directory
        if backup_dir is None:
            backup_dir = DATA_DIR / "backups"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"financial_data_backup_{timestamp}.db"
        else:
            backup_filename = "financial_data_backup.db"
        
        backup_path = backup_dir / backup_filename
        
        print(f"Creating database backup...")
        print(f"Source: {DATABASE_PATH}")
        print(f"Backup: {backup_path}")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create backup
        db_manager.backup_database(backup_path)
        
        # Verify backup
        backup_stats = DatabaseManager(backup_path).get_database_stats()
        original_stats = db_manager.get_database_stats()
        
        print(f"Backup completed successfully!")
        print(f"Backup size: {backup_stats['database_size_mb']:.2f} MB")
        print(f"Original size: {original_stats['database_size_mb']:.2f} MB")
        
        # Compare table counts
        if 'financial_data' in backup_stats['tables'] and 'financial_data' in original_stats['tables']:
            backup_rows = backup_stats['tables']['financial_data']['row_count']
            original_rows = original_stats['tables']['financial_data']['row_count']
            print(f"Records backed up: {backup_rows:,} / {original_rows:,}")
            
            if backup_rows != original_rows:
                print("Warning: Backup row count doesn't match original!")
                return False
        
        return True
        
    except Exception as e:
        print(f"Backup failed: {str(e)}")
        logging.error(f"Database backup failed: {str(e)}")
        return False


def backup_raw_data(backup_dir: Path = None) -> bool:
    """Backup raw data files.
    
    Args:
        backup_dir: Directory to store backup
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if backup_dir is None:
            backup_dir = DATA_DIR / "backups"
        
        raw_backup_dir = backup_dir / "raw_data"
        raw_backup_dir.mkdir(parents=True, exist_ok=True)
        
        from config.settings import RAW_DATA_DIR
        
        if not RAW_DATA_DIR.exists():
            print("No raw data directory found")
            return True
        
        # Get list of files to backup
        files_to_backup = list(RAW_DATA_DIR.glob("*"))
        files_to_backup = [f for f in files_to_backup if f.is_file()]
        
        if not files_to_backup:
            print("No raw data files found to backup")
            return True
        
        print(f"Backing up {len(files_to_backup)} raw data files...")
        
        for file_path in files_to_backup:
            destination = raw_backup_dir / file_path.name
            shutil.copy2(file_path, destination)
            print(f"  Backed up: {file_path.name}")
        
        print(f"Raw data backup completed!")
        print(f"Backup location: {raw_backup_dir}")
        
        return True
        
    except Exception as e:
        print(f"Raw data backup failed: {str(e)}")
        logging.error(f"Raw data backup failed: {str(e)}")
        return False


def cleanup_old_backups(backup_dir: Path, keep_count: int = 10) -> None:
    """Clean up old backup files, keeping only the most recent ones.
    
    Args:
        backup_dir: Directory containing backups
        keep_count: Number of backups to keep
    """
    try:
        if not backup_dir.exists():
            return
        
        # Get all backup files
        backup_files = list(backup_dir.glob("financial_data_backup_*.db"))
        
        if len(backup_files) <= keep_count:
            return
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old backups
        files_to_remove = backup_files[keep_count:]
        
        print(f"Cleaning up {len(files_to_remove)} old backup files...")
        
        for backup_file in files_to_remove:
            backup_file.unlink()
            print(f"  Removed: {backup_file.name}")
        
        print("Backup cleanup completed!")
        
    except Exception as e:
        print(f"Backup cleanup failed: {str(e)}")
        logging.error(f"Backup cleanup failed: {str(e)}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Backup financial analysis data')
    parser.add_argument('--backup-dir', help='Custom backup directory')
    parser.add_argument('--no-timestamp', action='store_true', help='Don\'t include timestamp in filename')
    parser.add_argument('--include-raw', action='store_true', help='Also backup raw data files')
    parser.add_argument('--cleanup', type=int, default=10, help='Number of backups to keep (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Financial Analysis Data Backup")
    print("=" * 40)
    
    # Set backup directory
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    
    success = True
    
    # Backup database
    db_success = backup_database(
        backup_dir=backup_dir,
        include_timestamp=not args.no_timestamp
    )
    success = success and db_success
    
    # Backup raw data if requested
    if args.include_raw:
        print("\n" + "-" * 40)
        raw_success = backup_raw_data(backup_dir)
        success = success and raw_success
    
    # Cleanup old backups
    if args.cleanup > 0:
        print("\n" + "-" * 40)
        cleanup_backup_dir = backup_dir if backup_dir else DATA_DIR / "backups"
        cleanup_old_backups(cleanup_backup_dir, args.cleanup)
    
    print("\n" + "=" * 40)
    if success:
        print("Backup completed successfully!")
        sys.exit(0)
    else:
        print("Backup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
