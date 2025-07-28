import sqlite3
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
DB_PATH = Path("C:/Projects/financial_summary/data/database/financial.db")
PROCESSED_DIR = Path("C:/Projects/financial_summary/data/processed")

# === CONNECT TO DATABASE ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === FUNCTION TO CHECK IF RECORD EXISTS ===
def record_exists(pc_number, date, account):
    query = """
    SELECT 1 FROM income_statement
    WHERE pc_number = ? AND statement_date = ? AND account = ?
    LIMIT 1
    """
    cursor.execute(query, (pc_number, date, account))
    return cursor.fetchone() is not None

# === MAIN UPLOAD LOOP ===
inserted_count = 0
skipped_count = 0

for file in PROCESSED_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"❌ Failed to read {file.name}: {e}")
        continue

    print(f"📂 Processing {file.name} ({len(df)} rows)")
    for _, row in df.iterrows():
        if record_exists(row["pc_number"], row["statement_date"], row["account"]):
            skipped_count += 1
            continue
        try:
            cursor.execute("""
                INSERT INTO income_statement (pc_number, statement_date, category, account, amount)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row["pc_number"],
                row["statement_date"],
                row["category"],
                row["account"],
                row["amount"]
            ))
            inserted_count += 1
        except Exception as e:
            print(f"⚠️ Failed to insert row: {e}")
            continue

    conn.commit()
    print(f"✅ Done: {file.name} — Inserted: {inserted_count}, Skipped: {skipped_count}")

conn.close()
print(f"\n🎯 Upload complete — Total Inserted: {inserted_count}, Skipped: {skipped_count}")
