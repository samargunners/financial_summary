import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date

# --- Configuration ---
DB_PATH = "C:/Projects/financial_summary/data/database/financial.db"

# --- Load Income Data ---
@st.cache_data(ttl=3600)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM income_statement", conn)
    conn.close()
    df["statement_date"] = pd.to_datetime(df["statement_date"])
    return df

df = load_data()

# --- Page Setup ---
st.set_page_config(page_title="📊 Financial Dashboard", layout="wide")
st.title("📊 Financial Performance Summary")

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔎 Filters")
    pc_numbers = sorted(df["pc_number"].unique().tolist())
    selected_store = st.selectbox("Select Store", pc_numbers)

# --- Filter by Store ---
store_df = df[df["pc_number"] == selected_store].copy()

# --- Date Calculations ---
today = date.today()
this_year = today.year
this_quarter = (today.month - 1) // 3 + 1

def filter_ytd(data, year):
    return data[
        (data["statement_date"].dt.year == year) &
        (data["statement_date"].dt.date <= today)
    ]

def filter_qtd(data, year, quarter):
    start_month = (quarter - 1) * 3 + 1
    start_date = date(year, start_month, 1)
    return data[
        (data["statement_date"].dt.date >= start_date) &
        (data["statement_date"].dt.date <= today)
    ]

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 YTD",
    "📆 QTD",
    "🔁 YTD Last Year",
    "🔁 QTD Last Year"
])

# --- Shared View Function ---
def show_summary(df_filtered, label):
    st.subheader(label)
    summary = df_filtered.groupby(["category", "account"])["amount"].sum().reset_index()
    st.dataframe(summary, use_container_width=True)

# --- Tab 1: Year-to-Date ---
with tab1:
    ytd_df = filter_ytd(store_df, this_year)
    show_summary(ytd_df, f"📅 Year-to-Date ({this_year})")

# --- Tab 2: Quarter-to-Date ---
with tab2:
    qtd_df = filter_qtd(store_df, this_year, this_quarter)
    show_summary(qtd_df, f"📆 Quarter-to-Date (Q{this_quarter} {this_year})")

# --- Tab 3: YTD Last Year ---
with tab3:
    ytd_last_year_df = filter_ytd(store_df, this_year - 1)
    show_summary(ytd_last_year_df, f"🔁 Year-to-Date Last Year ({this_year - 1})")

# --- Tab 4: QTD Last Year ---
with tab4:
    qtd_last_year_df = filter_qtd(store_df, this_year - 1, this_quarter)
    show_summary(qtd_last_year_df, f"🔁 Quarter-to-Date Last Year (Q{this_quarter} {this_year - 1})")
