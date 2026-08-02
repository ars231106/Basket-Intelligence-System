# ATLASQUANT — Basket Intelligence System

A quantitative portfolio construction and backtesting pipeline. It groups stocks into "baskets" using similarity graphs and community detection, scores them with either hardcoded rules or an ML ensemble, allocates capital across baskets and within baskets using several selectable strategies, and backtests the result with realistic walk-forward validation. A local Streamlit dashboard displays saved results.

This is not a RAG or LLM project — it's a classical quant/data-science system built on pandas, scikit-learn, NetworkX, and SciPy.

## How it works

1. **Universe selection** — choose one of 12 supported indices: NIFTY50, NIFTY100, S&P500, NASDAQ100, DOWJONES, DAX, FTSE100, HSI, NIFTYMID150, NIKKEI225, KOSPI200, NIFTYSMALL250.
2. **Data download** — pulls OHLCV price history via `yfinance` for every stock in the universe.
3. **Feature engineering** — computes 11 features per stock: Mean Return, Volatility, Momentum, RSI, ATR, Average Volume, Volume Growth, Distance from 20/50-day SMA, Sharpe Ratio, and 52-Week Range Position.
4. **Similarity graph + community detection** — builds a correlation-based similarity graph and runs Louvain community detection to find naturally clustered groups of stocks.
5. **Basket formation** — splits each community into baskets of 5–10 stocks.
6. **Stock scoring** — hardcoded weighted scoring, or an ML ensemble (Decision Tree + Random Forest + Gradient Boosting) trained on a walk-forward window with an automatic held-out evaluation step.
7. **Basket metrics, scoring, and ranking**.
8. **Basket selection** — pick the top `k` ranked baskets.
9. **Basket-level allocation** — choose one of 6 strategies: Equal Weight, Score Weighted, Inverse Volatility, Markowitz Optimization, Risk Parity, or Black-Litterman.
10. **Within-basket allocation** — choose one of 5 strategies for splitting each basket's capital among its own member stocks: Equal Weight, Range Position (mid-weighted), 52-Week High Momentum, Inverse Volatility (Intra-Basket), or Volume-Conditioned Momentum.
11. **Capital sizing check** — since most non-US exchanges (NSE/BSE included) require whole-share purchases, this computes the minimum capital actually needed to buy every selected stock as at least one share, and flags any stock that would get zero shares at your entered capital.
12. **Walk-forward backtest (optional)** — monthly rebalancing, realized forward returns, transaction costs, and comparison against an equal-weight-universe benchmark, reporting Total Return, CAGR, Volatility, Sharpe Ratio, Max Drawdown, and Turnover.

## Project structure

```
main.py                  Interactive end-to-end run: universe -> ... -> optional backtest
run_backtest.py           Standalone script for sweeping/comparing saved configurations
dashboard.py               Local Streamlit viewer over saved results (read-only, no pipeline logic)
train_ensemble_scores.py   Standalone ML ensemble training utility

src/
  download_data.py             Ticker normalization + yfinance download
  build_dataset.py, feature_engineering.py, feature_scaling.py
  similarity_engine.py, graph_builder.py, community_detection.py
  save_communities.py, visualisation.py
  stock_scoring.py, ml_scoring.py
  basket_generator.py, basket_metrics.py, basket_scoring.py
  portfolio_constructor.py, portfolio_allocator.py
  markowitz_optimiser.py, risk_parity_optimiser.py, black_litterman.py
  basket_returns.py
  intra_basket_allocator.py    Within-basket allocation strategies
  capital_sizing.py            Minimum viable capital / affordability check
  backtest_dataset.py, backtest_engine.py

universes/     Symbol,Name CSVs for each supported index
tests/         pytest suite (30 tests, each targeting a real bug found in this project)
.github/workflows/ci.yml   GitHub Actions: installs deps and runs pytest on every push/PR
```

Generated output (`data/`, `features/`, `graphs/`, `communities/`, `baskets/`, `portfolio/`, `reports/`, `backtests/`) is git-ignored — only source code and universe definitions are version-controlled.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

Run the full interactive pipeline:

```bash
python main.py
```

You'll be prompted step by step for: universe, date range, scoring method, number of baskets, capital, basket-level allocation strategy, within-basket allocation strategy, and (optionally) a backtest date range. Every stage's output is saved to disk as it runs.

To compare multiple saved configurations at once:

```bash
python run_backtest.py
```

To view results in a browser dashboard:

```bash
streamlit run dashboard.py
```

The dashboard is read-only — it never downloads data or runs the pipeline itself; it only displays whatever `main.py` / `run_backtest.py` have already saved.

## Testing & CI

```bash
pytest
```

Every push and pull request to `main` automatically installs dependencies and runs the full test suite via GitHub Actions (`.github/workflows/ci.yml`).

## Known limitations

- Backtests use a fixed basket/community structure computed once from full-period data, not re-clustered at every rebalance step.
- The backtest's ML model is trained once per run on a fixed window, not periodically retrained.
- Transaction costs are modeled as a flat 10bps of turnover, not volume- or spread-aware.
- `run_backtest.py` does not yet expose the within-basket strategy choice.
- Capital sizing assumes NSE/BSE-style whole-share trading constraints; markets that support fractional shares aren't affected by this limitation.
