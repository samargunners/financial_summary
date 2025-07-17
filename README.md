# Financial Analysis Dashboard

A comprehensive financial analysis tool for processing Excel data, analyzing financial metrics, and creating interactive dashboards.

## Features

- Excel file import and processing
- Financial metrics calculation
- Trend analysis
- Interactive dashboard with Dash/Plotly
- SQLite database for data storage
- Data cleaning and validation

## Project Structure

```
Financial-Analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── raw/               # Original Excel files
│   ├── processed/         # Cleaned data
│   └── database/          # SQLite database
├── src/                   # Source code
│   ├── data_processing/   # Data import and cleaning
│   ├── analysis/          # Financial analysis
│   └── dashboard/         # Dashboard application
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks for exploration
└── tests/                 # Unit tests
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up the database:
   ```bash
   python scripts/setup_database.py
   ```

3. Import your data:
   ```bash
   python scripts/import_data.py
   ```

4. Run the dashboard:
   ```bash
   python scripts/run_dashboard.py
   ```

## Usage

1. Place your Excel files in the `data/raw/` directory
2. Run the import script to process and clean the data
3. Use the Jupyter notebooks for data exploration
4. Launch the dashboard for interactive analysis

## Requirements

- Python 3.8+
- pandas
- plotly
- dash
- sqlite3
- openpyxl

## License

MIT License
