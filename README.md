# NYSE Screener (CSV-only)

Streamlit app that loads `nyse_finnhub_financials.csv`, computes a fundamentals-based factor score, and ranks tickers.
No live API calls.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
