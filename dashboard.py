import glob
import os

import pandas as pd
import streamlit as st

# Read-only viewer over whatever main.py / run_backtest.py has already
# saved to disk. This never downloads data, scores stocks, or runs a
# backtest itself -- it only opens the CSVs those scripts already wrote,
# so it stays a pure presentation layer with no pipeline logic duplicated
# here. Run with: streamlit run dashboard.py

st.set_page_config(page_title="ATLASQUANT Dashboard", layout="wide")
st.title("ATLASQUANT -- Basket Intelligence System")
st.caption("Run main.py or run_backtest.py from the terminal as usual -- this only displays "
           "results that are already sitting on disk.")

BACKTESTS_DIR = "backtests"
STOCK_ALLOC_DIR = "portfolio/stock_allocations"
CAPITAL_SIZING_DIR = "portfolio/capital_sizing"
COMMUNITIES_DIR = "communities"


def discover_universes():
    """
    Any universe with at least one saved backtest summary, stock
    allocation, or capital sizing file -- built from what's actually on
    disk rather than a hardcoded list, so a universe you haven't run yet
    just doesn't appear instead of causing a file-not-found error.
    """
    universes = set()
    for path in glob.glob(os.path.join(BACKTESTS_DIR, "*_backtest_summary.csv")):
        universes.add(os.path.basename(path).replace("_backtest_summary.csv", ""))
    for path in glob.glob(os.path.join(STOCK_ALLOC_DIR, "*_stock_allocation.csv")):
        universes.add(os.path.basename(path).replace("_stock_allocation.csv", ""))
    for path in glob.glob(os.path.join(CAPITAL_SIZING_DIR, "*_capital_sizing.csv")):
        universes.add(os.path.basename(path).replace("_capital_sizing.csv", ""))
    return sorted(universes)


universes = discover_universes()

if not universes:
    st.warning("No saved results found yet. Run main.py (or run_backtest.py) from the terminal "
               "first -- this dashboard only displays what's already been saved to backtests/ "
               "and portfolio/.")
    st.stop()

universe = st.sidebar.selectbox("Universe", universes)
st.sidebar.markdown("---")
st.sidebar.caption("This dashboard never triggers downloads, scoring, or backtests -- it only "
                    "reads files main.py / run_backtest.py already wrote.")

# ---------------------------------------------------------------------------
# Backtest summary
# ---------------------------------------------------------------------------
st.header("Backtest Summary")

summary_path = os.path.join(BACKTESTS_DIR, f"{universe}_backtest_summary.csv")
if os.path.exists(summary_path):
    summary = pd.read_csv(summary_path)
    st.dataframe(summary, width="stretch")
else:
    st.info(f"No backtest summary saved yet for {universe}.")

# ---------------------------------------------------------------------------
# Equity curves -- pick any saved runs for this universe and overlay them
# ---------------------------------------------------------------------------
st.header("Equity Curve")

curve_paths = sorted(glob.glob(os.path.join(BACKTESTS_DIR, f"{universe}_*_equity_curve.csv")))

if curve_paths:
    labels = {
        os.path.basename(p).replace(f"{universe}_", "").replace("_equity_curve.csv", ""): p
        for p in curve_paths
    }
    chosen_labels = st.multiselect("Compare runs", list(labels.keys()), default=list(labels.keys())[:2])

    if chosen_labels:
        combined = None
        for label in chosen_labels:
            df = pd.read_csv(labels[label])
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            series = df.set_index(date_col)["Equity"].rename(label)
            combined = series.to_frame() if combined is None else combined.join(series, how="outer")
        st.line_chart(combined)
    else:
        st.info("Pick at least one run above to plot.")
else:
    st.info(f"No equity curves saved yet for {universe}.")

# ---------------------------------------------------------------------------
# Per-stock allocation
# ---------------------------------------------------------------------------
st.header("Per-Stock Capital Allocation")

alloc_path = os.path.join(STOCK_ALLOC_DIR, f"{universe}_stock_allocation.csv")
if os.path.exists(alloc_path):
    allocation = pd.read_csv(alloc_path)
    st.dataframe(allocation, width="stretch")
    st.bar_chart(allocation.set_index("Symbol")["Capital_Distribution"])
else:
    st.info(f"No per-stock allocation saved yet for {universe}. Run main.py's within-basket "
            f"allocation step first.")

# ---------------------------------------------------------------------------
# Capital sizing check
# ---------------------------------------------------------------------------
st.header("Capital Sizing Check")

sizing_path = os.path.join(CAPITAL_SIZING_DIR, f"{universe}_capital_sizing.csv")
if os.path.exists(sizing_path):
    sizing = pd.read_csv(sizing_path)
    if not sizing.empty:
        floor_row = sizing.iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("Minimum viable capital", f"{floor_row['Required_Capital_For_1_Share']:,.2f}")
        col2.metric("Set by", f"{floor_row['Symbol']}", help=str(floor_row.get("Company", "")))
    st.dataframe(sizing, width="stretch")
else:
    st.info(f"No capital sizing check saved yet for {universe}.")

# ---------------------------------------------------------------------------
# Community structure
# ---------------------------------------------------------------------------
st.header("Community Structure")

community_path = os.path.join(COMMUNITIES_DIR, f"{universe}_communities.csv")
if os.path.exists(community_path):
    communities = pd.read_csv(community_path)
    st.dataframe(communities, width="stretch")
else:
    st.info(f"No community structure saved yet for {universe}.")
