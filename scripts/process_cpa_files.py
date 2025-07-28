from pathlib import Path
import pandas as pd
import re

# === CONFIGURATION ===
RAW_DIR = Path("C:/Projects/financial_summary/data/raw")
PROCESSED_DIR = Path("C:/Projects/financial_summary/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === CATEGORY HEADERS TO TRACK ===
category_headers = [
    "Sales",
    "Cost of Goods Sold",
    "Operating Expenses",
    "Other Income (Expenses)"
]

# === STORE NAME TO PC NUMBER MAP (cleaned version) ===
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

# === Extract standardized store name from file name ===
def extract_store_key(filename: str):
    match = re.search(r"\d{4}\s(.+?)\sIncome", filename)
    if match:
        key = match.group(1).strip().lower().replace("-", "").replace(" ", "")
        return key
    return None

# === Main Processing ===
all_cleaned_data = []

for file in RAW_DIR.glob("*.xlsx"):
    try:
        df = pd.read_excel(file, header=None)
    except Exception as e:
        print(f"❌ Failed to read {file.name}: {e}")
        continue

    store_key = extract_store_key(file.name)
    pc_number = store_map.get(store_key)

    if not pc_number:
        print(f"⚠️ Skipping {file.name} (store not recognized)")
        continue

    # Detect row with date headers
    date_row_idx = None
    for i in range(0, 10):
        if df.iloc[i].astype(str).str.contains(r"\d{2}/\d{2}/\d{2}").sum() >= 5:
            date_row_idx = i
            break

    if date_row_idx is None:
        print(f"⚠️ Could not find month header row in {file.name}")
        continue

    date_row = df.iloc[date_row_idx]
    month_cols = [
        i for i, val in enumerate(date_row)
        if isinstance(val, str) and re.match(r"\d{2}/\d{2}/\d{2}", val.strip()) and "total" not in val.lower()
    ]

    current_category = None
    parsed_rows = []

    for idx in range(date_row_idx + 1, len(df)):
        row = df.iloc[idx]
        first_cell = str(row[1]).strip() if pd.notna(row[1]) else ""

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
                except Exception:
                    continue

    if parsed_rows:
        clean_df = pd.DataFrame(parsed_rows)
        output_file = PROCESSED_DIR / f"{store_key}_processed.csv"
        clean_df.to_csv(output_file, index=False)
        all_cleaned_data.append(output_file.name)
        print(f"✅ Processed and saved: {output_file.name}")
    else:
        print(f"⚠️ No usable data found in: {file.name}")

print(f"\n🎯 Done! {len(all_cleaned_data)} file(s) processed.")
