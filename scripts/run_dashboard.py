"""Run dashboard script."""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.app import run_server
from config.settings import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG


def main():
    """Main function for running the dashboard."""
    parser = argparse.ArgumentParser(description='Run Financial Analysis Dashboard')
    parser.add_argument('--host', default=DASHBOARD_HOST, help=f'Host address (default: {DASHBOARD_HOST})')
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help=f'Port number (default: {DASHBOARD_PORT})')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine debug mode
    if args.no_debug:
        debug_mode = False
    elif args.debug:
        debug_mode = True
    else:
        debug_mode = DASHBOARD_DEBUG
    
    print("Financial Analysis Dashboard")
    print("=" * 40)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {debug_mode}")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 40)
    
    try:
        # Check if database exists and has data
        from src.data_processing.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        stats = db_manager.get_database_stats()
        
        if 'financial_data' in stats['tables']:
            row_count = stats['tables']['financial_data']['row_count']
            print(f"Database ready: {row_count:,} records found")
        else:
            print("Warning: No financial data found in database")
            print("Run 'python scripts/setup_database.py' and 'python scripts/import_data.py' first")
        
        print("Starting dashboard server...")
        
        # Run the dashboard
        run_server(
            host=args.host,
            port=args.port,
            debug=debug_mode
        )
        
    except KeyboardInterrupt:
        print("\nDashboard server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError starting dashboard: {str(e)}")
        logging.error(f"Dashboard startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
