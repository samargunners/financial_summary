from pathlib import Path
import pandas as pd
import re

# === CONFIGURATION ===
RAW_DIR = Path("C:/Projects/financial_summary/data/raw")
PROCESSED_DIR = Path("C:/Projects/financial_summary/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === KNOWN CATEGORY HEADERS ===
category_headers = [
    "Sales",
    "Cost of Goods Sold",
    "Operating Expenses",
    "Other Income (Expenses)"
]

# === MOCKED STORE MAP (to be loaded dynamically in actual script) ===
store_map = {
    "enola": "357993",
    "paxton": "301290",
    "mtjoy": "343939",
    "columbia": "358529",
    "lititz": "359042",
    "etown": "364322",
    "marietta": "363271",
    "eisenhower": "362913"
}

# === EXTRACT STORE NAME FROM FILE ===
def extract_store_key(filename):
    match = re.search(r"2025\s(.+?)\sIncome", filename)
    if match:
        return match.group(1).strip().lower().replace("-", "").replace(" ", "")
    return None

# === PROCESS EACH FILE ===
all_cleaned_data = []

for file in RAW_DIR.glob("*.xlsx"):
    df = pd.read_excel(file, header=None)
    store_key = extract_store_key(file.name)
    pc_number = store_map.get(store_key)

    if not pc_number:
        print(f"⚠️ Skipping {file.name} (store not recognized)")
        continue

    # Find the row with date headers
    date_row_idx = df[df.apply(lambda row: row.astype(str).str.contains(r"/\d{2}/\d{2}", regex=True).any(), axis=1)].index[0]
    date_row = df.iloc[date_row_idx]
    month_cols = [i for i, val in enumerate(date_row) if isinstance(val, pd.Timestamp) or isinstance(val, str)]

    current_category = None
    parsed_rows = []

    for idx in range(date_row_idx + 1, len(df)):
        row = df.iloc[idx]
        first_cell = str(row[0]).strip()

        if first_cell in category_headers:
            current_category = first_cell
            continue

        if current_category and first_cell and first_cell.lower() != "total":
            for col in month_cols:
                try:
                    date_val = pd.to_datetime(date_row[col])
                    amount = row[col]
                    if pd.notna(amount) and amount != 0:
                        parsed_rows.append({
                            "pc_number": pc_number,
                            "statement_date": date_val.strftime("%Y-%m-%d"),
                            "category": current_category,
                            "account": first_cell,
                            "amount": float(amount)
                        })
                except:
                    continue

    clean_df = pd.DataFrame(parsed_rows)
    output_file = PROCESSED_DIR / f"{store_key}_processed.csv"
    clean_df.to_csv(output_file, index=False)
    all_cleaned_data.append(clean_df)

# === Display preview of all processed data ===
combined_preview = pd.concat(all_cleaned_data).reset_index(drop=True)
import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned CPA Income Data", dataframe=combined_preview.head(100))
