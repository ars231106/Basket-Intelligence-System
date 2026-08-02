# ATLASQUANT — Basket Intelligence System
### Full project summary (for resume/portfolio drafting in a separate chat)

A personal, end-to-end quantitative portfolio construction and backtesting system built in Python from scratch. Clusters stocks into "baskets" via similarity graphs and community detection, scores them with hardcoded or ML-based logic, allocates capital across baskets using six different strategies and within baskets using five more, backtests the result with realistic walk-forward methodology, and displays everything in a local dashboard. Now on GitHub with a working CI pipeline and a 30-test pytest suite.

---

## 1. Pipeline architecture (12 stages, in order)

1. **Universe selection** — 12 supported indices/markets: NIFTY50, NIFTY100, S&P500, NASDAQ100, DOWJONES, DAX, FTSE100, HSI, NIFTYMID150, NIKKEI225, KOSPI200, NIFTYSMALL250. Spans US, Indian, German, UK, Hong Kong, Japanese, and Korean equity markets.
2. **Data ingestion** — downloads OHLCV price history per stock via `yfinance`. Custom ticker-normalization logic handles: US share-class dots vs. hyphens (`BRK.B`→`BRK-B`), international exchange suffixes (`.NS`, `.BO`, `.DE`, `.L`, `.HK`, `.PA`, `.MI`, `.TO`, `.AX`, `.SW`, `.T`, `.KS`), LSE-specific slash conventions (EPIC dots, share-class separators), Windows-reserved filename collisions (e.g. `CON.DE.csv` silently mangled by Windows/OneDrive), and filesystem-unsafe characters.
3. **Feature engineering** — 11 computed features per stock: Mean Return, Volatility, Momentum, RSI, ATR, Average Volume, Volume Growth, Distance from 20/50-day SMA, Sharpe Ratio, and a custom **52-week range position** feature (0 = at 52-week low, 1 = at 52-week high) built specifically to support the within-basket strategies below.
4. **Similarity graph + community detection** — builds a correlation-based similarity graph across all stocks in a universe, then runs **Louvain community detection** (via `networkx` + `python-louvain`) to discover natural groupings of similarly-behaving stocks — pure structure discovery, no return/risk judgment yet.
5. **Basket formation** — splits each community into appropriately sized "baskets" (5–10 stocks each) for portfolio construction.
6. **Stock scoring** — two selectable methods:
   - **Hardcoded weighted scoring** (deterministic, feature-based)
   - **ML ensemble**: Decision Tree + Random Forest + Gradient Boosting, blended with hardcoded weights, trained on a proper walk-forward window with an automatic 80/20 chronological held-out evaluation step (reports Pearson correlation, rank correlation, and top-minus-bottom quantile spread) before training final models on the full range for live use.
7. **Basket metrics + scoring + ranking** — aggregates member-stock statistics per basket and produces a ranked list.
8. **Basket selection** — user picks how many top-ranked baskets (`k`) to include in the portfolio.
9. **Basket-level capital allocation** — 6 strategies, split into heuristic and optimization-based:
   - Equal Weight, Score Weighted, Inverse Volatility (heuristic)
   - **Markowitz mean-variance optimization**, **Risk Parity**, **Black-Litterman** (full Bayesian implementation: equilibrium returns via a volume-based market-weight proxy, a picking matrix, standardized-score-based view returns, and a properly-scaled Omega confidence matrix)
10. **Within-basket (intra-basket) capital allocation** — a second, independent allocation layer deciding how each basket's own capital splits among its member stocks. 5 strategies:
    - Equal Weight (baseline)
    - **Range Position** (mid-weighted — heaviest weight to stocks sitting in the middle of their 52-week trading range)
    - **52-Week High Momentum** (heaviest weight to stocks near their 52-week high with strong momentum — grounded in George & Hwang, 2004, *Journal of Finance*)
    - **Inverse Volatility (intra-basket)** — same logic as the basket-level version, applied one level down
    - **Volume-Conditioned Momentum** (momentum winners discounted by recent volume spikes — grounded in Lee & Swaminathan, 1998)
11. **Capital sizing / affordability check** — computes the exact minimum total capital required to buy every selected stock as at least one whole share close to its target weight (formula: `price ÷ target weight`, not price alone), and flags which specific stocks would receive zero shares at a given capital level — addressing the real-world constraint that NSE/BSE (and most non-US exchanges) don't support fractional share trading.
12. **Walk-forward backtesting engine** — monthly rebalancing, realized (not estimated) forward returns computed from actual entry/exit prices, transaction costs (turnover × 10bps), equal-weight-universe benchmark comparison, and full summary statistics: Total Return, CAGR, Annualized Volatility, Sharpe Ratio, Max Drawdown, Average Turnover, Number of Rebalances.

**Plus:** a local **Streamlit dashboard** (`dashboard.py`) — a read-only viewer over saved results with a universe dropdown, overlay-able equity curve comparison across multiple saved runs, per-stock allocation table/chart, capital sizing summary, and community structure browser. Deliberately built as pure Python (no HTML/CSS/JS, no database, no API layer) to stay firmly a data-science project rather than a web-dev one.

---

## 2. Real bugs found and fixed (via empirical validation against real data, not just code review)

1. **Black-Litterman Omega/tau scale mismatch** — the model's "confidence in a view" matrix (Omega) lived on an arbitrary fixed 1–20 scale, completely disconnected from `tau × Sigma`'s actual variance units. Result: posterior returns were numerically indistinguishable from equilibrium returns regardless of the confidence/sensitivity parameter — views had zero effect. Diagnosed by building a scipy/sklearn-free numpy replica of the model to isolate the issue, then fixed by rescaling Omega to `(tau × variance) / confidence`. Verified: posterior now correctly equals equilibrium at zero sensitivity and diverges meaningfully at higher sensitivity.
2. **Basket_ID / weight positional misalignment bug** — `covariance_matrix.loc[selected_ids, selected_ids]` reorders to rank order via pandas `.loc`, but a parallel `.isin()` filter elsewhere only filtered without reordering, causing basket weights to be silently mismatched to the wrong Basket_IDs downstream — affecting Markowitz, Risk Parity, and Black-Litterman output labeling, and the Markowitz objective function itself. Verified by reproducing the exact scrambled pattern in a correctly-labeled replica and matching it byte-for-byte against the real (mislabeled) saved output. Fixed via `.set_index("Basket_ID").loc[selected_ids].reset_index()`.
3. **Portfolio allocator string-mismatch bug** — `main.py`'s strategy map produced `"equal_weighted"` for menu option 1, but `portfolio_allocator.py` only recognized `"equally_weighted"` — meaning the literal first menu option in the entire system had always thrown an unhandled `ValueError`. Caught incidentally while restoring accidentally-stripped comments during an unrelated file-sync fix; disclosed and fixed as a clean one-line change.
4. **KOSPI ticker-suffix bug** — `KNOWN_EXCHANGE_SUFFIXES` (the set of exchange codes exempted from US share-class dot-to-hyphen conversion) never included `"KS"` (Korea), since KOSPI was the first Korean universe ever added. Result: every single ticker like `000080.KS` was silently mangled to `000080-KS` before hitting Yahoo Finance, causing **100% of a 200-stock download to fail**. Diagnosed by noticing the error message showed a hyphenated symbol instead of the dotted original. Fixed by adding `"KS"` to the set.
5. **ML train/test split not wired into live scoring** — caught that the "walk-forward" scoring path was training models on the entire user-specified date range with no actual held-out evaluation, contradicting the stated purpose. Fixed by adding an automatic 80/20 chronological held-out evaluation step that reports real generalization metrics before training the final live model.

---

## 3. Data engineering / data quality work

- **Nikkei 225 universe**: cleaned a raw pasted CSV of 225 Japanese companies (all-caps names) into a clean `Symbol,Name` format with proper title casing and `.T` ticker suffixes. Built a custom title-casing algorithm handling acronym preservation (NTT, KDDI, UFJ, ANA, SoftBank, SUMCO), hyphenated and dotted company names, and one manual correction for source data truncated mid-name (Tokyo Electric Power / TEPCO). Discovered and replaced a pre-existing but stale, wrong-schema `nikkei225.csv` already sitting in the project.
- **KOSPI 200 universe**: filtered a 3,518-row full Korean market listing down to the true 200-member index by cross-referencing against Wikipedia's current official KOSPI 200 constituent table, since the user's uploaded snapshot was missing 24 current constituents (companies that IPO'd or restructured since ~2019, e.g. LG Energy Solution, Krafton, KakaoBank). Flagged and correctly preserved one legitimately unusual alphanumeric ticker (Samsung Epis Holdings, `0126Z0`).
- **Corruption handling**: diagnosed trailing null-byte corruption in several saved CSVs, traced to interrupted/partial OneDrive sync writes — worked around via `.notna()` filtering rather than silently "fixing" the pipeline.
- **Download failure diagnosis**: correctly distinguished a transient Yahoo Finance connection interruption (a contiguous block of ~30 failed tickers, self-recovering) from a genuine, deterministic code bug (100% failure across every Korean ticker) by reading the actual error text rather than assuming both were the same class of problem.

---

## 4. Software engineering / DevOps (built from literally zero this session)

- **Git & GitHub from scratch**: installed Git for Windows, first-time global config, created and connected a GitHub repository, resolved an unrelated-histories merge conflict from a pre-existing remote commit.
- **`.gitignore` design**: excluded all regenerable output (`venv/`, `data/`, `features/`, `graphs/`, `communities/`, `baskets/`, `portfolio/`, `reports/`, `backtests/`) — kept only source code under version control, a deliberate "structure vs. output" separation.
- **`requirements.txt`**: generated via `pip freeze` for a fully reproducible environment.
- **pytest test suite — 30 tests, all targeting real bugs found in this project, not generic coverage**:
  - 7 tests on ticker normalization (the KOSPI bug)
  - 4 tests on the allocator strategy-string bug
  - 4 tests on Black-Litterman Omega scaling
  - 10 tests on the four within-basket allocation strategies + dispatcher
  - 5 tests on the capital sizing / affordability module
- **GitHub Actions CI pipeline** (`.github/workflows/ci.yml`), built incrementally from a single job to a production-shaped setup:
  - Two **parallel** jobs: `lint` (via `ruff`, deliberately scoped to only real-bug-catching rules — syntax errors and undefined-name/unused-import checks — not cosmetic style) and `test`
  - **OS matrix**: tests run on both `ubuntu-latest` and `windows-latest` — deliberately included Windows because of real, previously-found Windows/OneDrive-specific bugs (the reserved-filename mangling issue) that a Linux-only runner would never catch
  - `fail-fast: false` so both OS results are always visible, not just the first failure
  - **pip dependency caching** for faster runs
  - **Concurrency group with cancel-in-progress** to avoid wasting CI time on superseded pushes
- **Debugging a Windows PowerShell encoding gotcha**: diagnosed that plain `pip freeze > requirements.txt` redirection can produce UTF-16-encoded output on Windows PowerShell, and separately did a rigorous multi-step investigation of a `git diff` "binary files differ" false-positive (systematically ruled out UTF-16 encoding, a UTF-8 BOM, `.gitattributes` overrides, and full-file null-byte corruption via direct byte-level inspection) before concluding it was a cosmetic diff-display quirk with no actual effect on file integrity — verified via a passing CI run.

---

## 5. Real analytical findings from backtests across universes

- **Risk Parity and Inverse Volatility** (risk-based allocation) beat their equal-weight-universe benchmark on return, volatility, *and* max drawdown simultaneously — on every universe tested (FTSE100, S&P500).
- **Black-Litterman** consistently beat the benchmark on return and Sharpe ratio but showed a deeper max drawdown (niftymid150 ×2, nifty100).
- **Markowitz and even plain Equal Weight** showed the same "wins on return/Sharpe, loses on drawdown" pattern specifically when capital was concentrated into very few baskets (DAX, Dow Jones).
- Developed a testable working hypothesis from these patterns: **diversification level (number of baskets held) may drive risk-adjusted outperformance more than the specific allocation philosophy used** — directly falsifiable using the backtest infrastructure built.
- Example real result: KOSPI 200, ML ensemble scoring + Black-Litterman allocation — **33.4% CAGR, 2.19 Sharpe, -8.4% max drawdown**, vs. benchmark's 27.0% CAGR, 2.12 Sharpe, -7.6% drawdown.

---

## 6. Real-world / practitioner-grade reasoning demonstrated

- Identified and correctly diagnosed the **fractional-share execution gap**: the backtest's weighted-return math implicitly assumes infinitely divisible capital, which doesn't reflect NSE/BSE's whole-share-only trading — grounded the discussion in the real industry concept of strategy **"capacity"** (the capital range where live execution tracks the idealized backtest) and named the standard practitioner solution pattern (discrete/greedy share-allocation algorithms, e.g. PyPortfolioOpt's `DiscreteAllocation`).
- Correctly reasoned through *why* using different allocation methodologies across baskets vs. within baskets is legitimate, not inconsistent — grounded in real institutional practice: **Hierarchical Risk Parity** (López de Prado), and **"top-down risk budgeting, bottom-up security selection"** as practiced by real endowments (the Yale model) and multi-strategy hedge funds.
- Grounded all four within-basket strategies in specific, cited academic finance literature rather than intuition alone: George & Hwang (2004, *Journal of Finance*) on 52-week-high momentum, Lee & Swaminathan (1998) on volume-conditioned momentum, Ang/Hodrick/Xing/Zhang (2006) on the low-volatility anomaly.
- Explicitly distinguished the project as a legitimate **data pipeline** (staged ETL: ingestion → transformation → output) from a **CI/CD pipeline** (which it lacked entirely until this session's DevOps work).

---

## 7. Known, disclosed limitations (worth being ready to discuss honestly in an interview)

- Backtest uses a **fixed basket/community structure** computed once from full-period correlation data, not re-clustered at every walk-forward step — a deliberate, disclosed simplification (full re-clustering would be far more expensive and less stable).
- The backtest's ML model is trained **once** per run on a fixed window, not periodically retrained as the walk-forward loop progresses.
- Transaction cost model is a **flat 10bps assumption**, not volume- or spread-aware.
- `main.py` is currently a single ~450-line top-level script rather than modular callable functions — the reason the Basket_ID alignment bug (now fixed) still can't be unit-tested directly, and the natural next refactor.
- `run_backtest.py` (the standalone "compare everything" script) hasn't yet been updated to expose the within-basket strategy choice added this session.
- No README exists yet — identified as the single highest-leverage missing piece before the project is fully "resume ready," since none of the above — the bug fixes, the academic grounding, the real backtest results — currently live anywhere but this chat history.

---

## 8. Skills this project demonstrates

**Quant finance / modeling:** Markowitz mean-variance optimization, Risk Parity, Black-Litterman (Bayesian posterior returns, equilibrium returns, confidence matrices), Sharpe ratio / CAGR / max drawdown / turnover computation, momentum and volatility factor research, walk-forward backtesting methodology, transaction cost modeling, capital allocation theory across a portfolio hierarchy.

**Machine learning:** ensemble modeling (Decision Tree + Random Forest + Gradient Boosting), proper train/test walk-forward validation, held-out evaluation metrics (rank correlation, quantile spread).

**Data science / engineering:** feature engineering from raw OHLCV data, graph theory and network analysis (similarity graphs, Louvain community detection via NetworkX), large-scale data cleaning across inconsistent international data sources, ETL pipeline design across 12 global equity markets.

**Software engineering:** Python (pandas, numpy, scipy, scikit-learn, networkx, matplotlib), git/GitHub from first principles, CI/CD pipeline design (GitHub Actions, OS matrix testing, parallel jobs, caching, concurrency control), test-driven bug fixing (pytest, 30 tests each targeting a real found defect), dependency management, cross-platform debugging (Windows-specific path/encoding issues).

**Applied research / analytical rigor:** grounding design decisions in cited academic literature, empirically validating mathematical models against real data rather than trusting plausible-looking output, systematic root-cause debugging (ruling out hypotheses one at a time with direct evidence), honest disclosure of methodology limitations and tradeoffs.

**Data visualization:** Streamlit dashboard development, deliberately scoped to stay a data-science artifact rather than a web-development one.
