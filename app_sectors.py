import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NYSE Screener (CSV-only)", layout="wide")
st.title("NYSE Screener (CSV-only)")
st.caption("Reads nyse_finnhub_financials.csv offline. No live API calls.")

CSV_PATH = "nyse_finnhub_financials.csv"

# -------------------- Helpers --------------------
@st.cache_data(ttl=60 * 60)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def winsorize(s: pd.Series, p=0.01) -> pd.Series:
    s = s.copy()
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def zscore_series(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd

def z_by_group(df: pd.DataFrame, col: str, group: str, enabled: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    if enabled and group in df.columns:
        return df.groupby(group)[col].transform(lambda s: zscore_series(s))
    return zscore_series(df[col])

def build_score(df: pd.DataFrame, sector_neutral: bool = True, winsor_p: float = 0.01) -> pd.DataFrame:
    """
    Uses YOUR Finnhub columns.
    Produces:
      - score (composite)
      - coverage (how much core data is present)
    Sector-neutral z-scores are applied if sector_neutral=True and 'sector' exists.
    """
    df = df.copy()

    # Ensure ticker column
    if "ticker" not in df.columns:
        if "symbol" in df.columns:
            df["ticker"] = df["symbol"].astype(str)
        else:
            df["ticker"] = df.index.astype(str)

    # Clean sector/industry strings if present
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).replace({"nan": np.nan}).str.strip()
    if "industry" in df.columns:
        df["industry"] = df["industry"].astype(str).replace({"nan": np.nan}).str.strip()

    # Core factor columns (your exact names)
    core_cols = [
        "metric_roaTTM",
        "metric_operatingMarginTTM",
        "metric_netProfitMarginTTM",
        "metric_roiTTM",
        "metric_beta",
        "metric_currentRatioAnnual",
        "metric_totalDebt/totalEquityAnnual",
        "metric_netInterestCoverageTTM",
        "metric_evEbitdaTTM",
        "metric_peTTM",
        "metric_pegTTM",
        "metric_marketCapitalization",
    ]

    # Optional columns (not required)
    optional_cols = [
        "metric_quickRatioAnnual",
        "metric_forwardPE",
        "metric_psTTM",
        "metric_pb",
        "metric_evRevenueTTM",
        "metric_revenueGrowthTTMYoy",
        "metric_epsGrowthTTMYoy",
        "metric_26WeekPriceReturnDaily",
        "metric_52WeekPriceReturnDaily",
        "metric_priceRelativeToS&P50026Week",
        "metric_priceRelativeToS&P50052Week",
    ]

    # Coerce numerics
    for c in core_cols + optional_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Winsorize numeric factor columns (robustness)
    for c in core_cols + optional_cols:
        if c in df.columns:
            df[c] = winsorize(df[c], winsor_p)

    # Inversions (lower is better)
    df["inv_beta"] = -df["metric_beta"] if "metric_beta" in df.columns else np.nan
    df["inv_debt"] = -df["metric_totalDebt/totalEquityAnnual"] if "metric_totalDebt/totalEquityAnnual" in df.columns else np.nan
    df["inv_ev_ebitda"] = -df["metric_evEbitdaTTM"] if "metric_evEbitdaTTM" in df.columns else np.nan
    df["inv_pe"] = -df["metric_peTTM"] if "metric_peTTM" in df.columns else np.nan
    df["inv_peg"] = -df["metric_pegTTM"] if "metric_pegTTM" in df.columns else np.nan

    # Sector-neutral scoring if sector exists
    group_col = "sector"
    use_group = sector_neutral and (group_col in df.columns) and df[group_col].notna().any()
    z = lambda col: z_by_group(df, col, group_col, use_group)

    # Composite fundamentals score
    df["score"] = (
        # Quality (40%)
        z("metric_roaTTM") * 0.16 +
        z("metric_operatingMarginTTM") * 0.10 +
        z("metric_netProfitMarginTTM") * 0.06 +
        z("metric_roiTTM") * 0.08 +

        # Risk (35%)
        z("inv_debt") * 0.14 +
        z("metric_netInterestCoverageTTM") * 0.10 +
        z("metric_currentRatioAnnual") * 0.05 +
        z("inv_beta") * 0.06 +

        # Valuation (25%)
        z("inv_ev_ebitda") * 0.14 +
        z("inv_pe") * 0.07 +
        z("inv_peg") * 0.04
    )

    # Coverage: fraction of core metrics present (excluding market cap)
    coverage_cols = [c for c in core_cols if c != "metric_marketCapitalization" and c in df.columns]
    df["coverage"] = df[coverage_cols].notna().mean(axis=1) if coverage_cols else 0.0

    return df

# -------------------- Load --------------------
try:
    raw = load_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"Missing {CSV_PATH}. Put it in the repo root next to app.py.")
    st.stop()

# Ensure ticker col
df = raw.copy()
if "ticker" not in df.columns:
    if "symbol" in df.columns:
        df["ticker"] = df["symbol"].astype(str)
    else:
        st.error("CSV must contain 'symbol' or 'ticker'.")
        st.stop()

# Build scores
st.sidebar.header("Scoring")
sector_neutral = st.sidebar.checkbox("Sector-neutral z-scores", value=True)
winsor_p = st.sidebar.slider("Winsorize (outlier cap)", 0.0, 0.05, 0.01, 0.005)

scored = build_score(df, sector_neutral=sector_neutral, winsor_p=winsor_p)

# -------------------- Filters --------------------
st.sidebar.header("Filters")
top_n = st.sidebar.slider("Show Top N", 10, 300, 50, 10)
min_coverage = st.sidebar.slider("Min data coverage", 0.0, 1.0, 0.60, 0.05)

# Apply coverage
scored = scored[scored["coverage"] >= min_coverage].copy()

# Market cap
min_mcap = st.sidebar.number_input("Min market cap ($)", value=5_000_000_000, step=1_000_000_000)
if "metric_marketCapitalization" in scored.columns:
    scored = scored[scored["metric_marketCapitalization"].fillna(0) >= min_mcap].copy()
else:
    st.sidebar.info("No metric_marketCapitalization column found; market cap filter skipped.")

# Sector filter
if "sector" in scored.columns and scored["sector"].notna().any():
    sectors = sorted(scored["sector"].dropna().unique().tolist())
    chosen_sectors = st.sidebar.multiselect("Sector", sectors, default=[])
    if chosen_sectors:
        scored = scored[scored["sector"].isin(chosen_sectors)].copy()
else:
    chosen_sectors = []

# Industry filter
if "industry" in scored.columns and scored["industry"].notna().any():
    industries = sorted(scored["industry"].dropna().unique().tolist())
    chosen_industries = st.sidebar.multiselect("Industry", industries, default=[])
    if chosen_industries:
        scored = scored[scored["industry"].isin(chosen_industries)].copy()
else:
    chosen_industries = []

# Risk filters
max_debt = st.sidebar.slider("Max TotalDebt/TotalEquity (Annual)", 0.0, 10.0, 2.0, 0.1)
min_cov = st.sidebar.slider("Min Net Interest Coverage (TTM)", 0.0, 50.0, 3.0, 0.5)
min_cr = st.sidebar.slider("Min Current Ratio (Annual)", 0.0, 10.0, 1.0, 0.1)
max_beta = st.sidebar.slider("Max Beta", 0.0, 5.0, 1.5, 0.1)

if "metric_totalDebt/totalEquityAnnual" in scored.columns:
    scored = scored[(scored["metric_totalDebt/totalEquityAnnual"].isna()) |
                    (scored["metric_totalDebt/totalEquityAnnual"] <= max_debt)]
if "metric_netInterestCoverageTTM" in scored.columns:
    scored = scored[(scored["metric_netInterestCoverageTTM"].isna()) |
                    (scored["metric_netInterestCoverageTTM"] >= min_cov)]
if "metric_currentRatioAnnual" in scored.columns:
    scored = scored[(scored["metric_currentRatioAnnual"].isna()) |
                    (scored["metric_currentRatioAnnual"] >= min_cr)]
if "metric_beta" in scored.columns:
    scored = scored[(scored["metric_beta"].isna()) |
                    (scored["metric_beta"] <= max_beta)]

# Rank
scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
view = scored.head(top_n)

# -------------------- Display --------------------
st.subheader("Ranked results")

display_cols = [
    "ticker",
    "score",
    "coverage",
    "sector",
    "industry",
    "metric_marketCapitalization",
    "metric_beta",
    "metric_evEbitdaTTM",
    "metric_peTTM",
    "metric_pegTTM",
    "metric_roaTTM",
    "metric_roiTTM",
    "metric_operatingMarginTTM",
    "metric_netProfitMarginTTM",
    "metric_totalDebt/totalEquityAnnual",
    "metric_netInterestCoverageTTM",
    "metric_currentRatioAnnual",
    "asof_utc",
]
display_cols = [c for c in display_cols if c in view.columns]

st.dataframe(view[display_cols], use_container_width=True)

# -------------------- Charts --------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Score distribution")
    fig = px.histogram(scored.dropna(subset=["score"]), x="score")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Score vs Beta")
    if "metric_beta" in scored.columns:
        fig2 = px.scatter(scored, x="metric_beta", y="score", hover_data=["ticker"] + (["sector"] if "sector" in scored.columns else []))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No metric_beta column in CSV.")

# Sector chart
if "sector" in scored.columns and scored["sector"].notna().any():
    st.subheader("Top scores by sector (median of top 50 per sector)")
    tmp = scored.dropna(subset=["sector", "score"]).copy()
    tmp["rank_in_sector"] = tmp.groupby("sector")["score"].rank(ascending=False, method="first")
    top50 = tmp[tmp["rank_in_sector"] <= 50]
    agg = top50.groupby("sector")["score"].median().sort_values(ascending=False).reset_index()
    fig3 = px.bar(agg, x="sector", y="score")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- Download --------------------
st.subheader("Download")
st.download_button(
    "Download filtered rankings (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_rankings.csv",
    mime="text/csv",
)

with st.expander("Show CSV columns"):
    st.write(list(raw.columns))
