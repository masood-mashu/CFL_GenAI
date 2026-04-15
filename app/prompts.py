def generate_prompt(metrics):
    return f"""
You are a senior business analyst.

Analyze this forecasting performance:

MAPE: {metrics['mape']:.2f}%
Over-forecast count: {metrics['over_forecast_count']}
Under-forecast count: {metrics['under_forecast_count']}

Worst performing products:
{metrics['worst_products']}

Generate a professional report with:

1. Executive Summary
2. Key Issues
3. Root Cause Analysis
4. Recommendations to improve forecasting

Keep it business-focused and concise.
"""