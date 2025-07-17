"""Application settings and configuration."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

# Database configuration
DATABASE_NAME = "financial_data.db"
DATABASE_PATH = DATABASE_DIR / DATABASE_NAME

# Dashboard configuration
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# File processing settings
SUPPORTED_FILE_TYPES = [".xlsx", ".xls", ".csv"]
MAX_FILE_SIZE_MB = 100

# Analysis settings
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
CURRENCY_SYMBOL = "$"

# Chart settings
DEFAULT_CHART_THEME = "plotly_white"
DEFAULT_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Data validation settings
MIN_DATE_YEAR = 2000
MAX_DATE_YEAR = 2030
REQUIRED_COLUMNS = ["date", "amount", "category"]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
