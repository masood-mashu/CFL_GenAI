from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
import requests

# ---------------- ENV ----------------
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# ---------------- FORECAST ----------------
FORECAST_BY_RANK = {
    1: 14754, 2: 62693, 3: 6576, 4: 9788, 5: 42000,
    6: 755, 7: 1968, 8: 2868, 9: 2463, 10: 591,
    11: 252, 12: 4259, 13: 562, 14: 5062, 15: 265,
    16: 1906, 17: 1529, 18: 849, 19: 8123, 20: 474,
    21: 216, 22: 12800, 23: 5400, 24: 2323, 25: 3209,
    26: 1304, 27: 1782, 28: 893, 29: 395, 30: 1221,
}

# ---------------- FLATTEN ----------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()

    flattened = []
    for column in df.columns:
        parts = []
        for part in column:
            text = str(part).strip()
            if text and not text.startswith("Unnamed"):
                parts.append(text)
        flattened.append(" ".join(parts) if parts else "")

    result = df.copy()
    result.columns = flattened
    return result

# ---------------- DATA PIPELINE ----------------
def build_cfl_dataset(excel_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name="Data Pack - Actual Bookings", header=[0, 1])

    column_lookup = {str(column[0]).strip(): column for column in raw.columns}
    rank_col = column_lookup["Cost Rank"]
    product_col = column_lookup["Product Name"]
    plc_col = column_lookup["Product Life Cycle"]
    actual_cols = [col for col in raw.columns if str(col[0]).strip() == "ACTUAL UNITS"]

    bookings = raw.loc[pd.to_numeric(raw[rank_col], errors="coerce").notna()].copy()
    bookings = bookings.loc[~bookings[rank_col].duplicated(keep="first")].copy()

    bookings["rank"] = pd.to_numeric(bookings[rank_col], errors="coerce").astype(int)
    bookings["product"] = bookings[product_col].astype(str).str.strip()
    bookings["plc"] = bookings[plc_col].astype(str).str.strip()

    def latest_actual(row):
        values = row[actual_cols].dropna()
        return int(values.iloc[-1]) if not values.empty else 0

    bookings["actual"] = bookings.apply(latest_actual, axis=1)
    bookings["forecast"] = bookings["rank"].map(FORECAST_BY_RANK)

    result = bookings[["rank", "product", "actual", "forecast", "plc"]].sort_values("rank")
    result = result.dropna(how="all")

    result.to_csv(output_path, index=False)
    return result

# ---------------- ANALYSIS ----------------
def calculate_metrics(df):
    df = _flatten_columns(df)

    df["error"] = df["forecast"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["ape"] = df["abs_error"] / df["actual"].replace(0, 1)

    mape = float(df["ape"].mean() * 100)

    worst_df = df.sort_values("ape", ascending=False).head(5)

    worst_products = [
        {"product": str(product), "ape": float(ape)}
        for product, ape in worst_df[["product", "ape"]].itertuples(index=False, name=None)
    ]

    return {
        "mape": mape,
        "over_forecast_count": int((df["error"] > 0).sum()),
        "under_forecast_count": int((df["error"] < 0).sum()),
        "worst_products": worst_products,
    }

# ---------------- PROMPT ----------------
def generate_prompt(metrics):
    return f"""
You are a business analyst.

Forecast Performance:
MAPE: {metrics['mape']:.2f}%
Over Forecast Count: {metrics['over_forecast_count']}
Under Forecast Count: {metrics['under_forecast_count']}

Worst products:
{metrics['worst_products']}

Generate:
1. Executive summary
2. Key issues
3. Recommendations

Keep it concise and professional.
"""

# ---------------- NVIDIA API ----------------
def call_nvidia_llm(prompt):
    if not NVIDIA_API_KEY:
        raise ValueError("❌ NVIDIA API key not found")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        print("📡 Calling NVIDIA (correct endpoint)...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("❌ API Error:", e)
        print("Response:", response.text if 'response' in locals() else "")
        return "Failed to generate report."

# ---------------- RUN ----------------
def run():
    print("🚀 Running CFL Insight Engine...\n")

    base_dir = Path(__file__).resolve().parents[1]
    excel_path = base_dir / "data" / "CFL_External Data Pack_Phase1.xlsx"
    csv_path = base_dir / "data" / "cfl_data.csv"

    print("📂 Excel exists:", excel_path.exists())

    if not excel_path.exists():
        raise FileNotFoundError(f"❌ Excel file not found: {excel_path}")

    df = build_cfl_dataset(excel_path, csv_path)
    print(f"✅ Dataset created: {len(df)} rows")

    metrics = calculate_metrics(df)
    print("📊 Metrics:", metrics)

    prompt = generate_prompt(metrics)

    print("\n=== AI REPORT ===\n")
    report = call_nvidia_llm(prompt)
    print(report)

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    run()