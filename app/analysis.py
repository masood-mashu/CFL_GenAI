import pandas as pd

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

def calculate_metrics(df):
    df = _flatten_columns(df)

    df["error"] = df["forecast"] - df["actual"]
    df["abs_error"] = abs(df["error"])
    df["ape"] = df["abs_error"] / df["actual"]
    
    mape = df["ape"].mean() * 100
    
    over = (df["error"] > 0).sum()
    under = (df["error"] < 0).sum()
    
    # Top errors
    worst = df.sort_values("ape", ascending=False).head(5)[["product", "ape"]]
    worst_products = [
        {"product": str(product), "ape": float(ape)}
        for product, ape in worst.itertuples(index=False, name=None)
    ]
    
    return {
        "mape": mape,
        "over_forecast_count": over,
        "under_forecast_count": under,
        "worst_products": worst_products
    }