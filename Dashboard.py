# Updated Portfolio.py with integrated fixes (risk parity patch)
# (Full file replacement)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================
# 1) USER CONFIG / PORTFOLIO
# ==========================
st.set_page_config(page_title="Quant Portfolio Dashboard", layout="wide")
st.title("Quant Portfolio Dashboard — Rebalance & Backtest")

# --- Portfolio definition (edit quantities / cost_basis here) ---
portfolio = {
    "AAPL": {"quantity": 4, "cost_basis": 278.03},
    "MSFT": {"quantity": 2, "cost_basis": 483.47},
    "ORCL": {"quantity": 1, "cost_basis": 198.85},
    "NVDA": {"quantity": 3, "cost_basis": 180.93},
    "ACHR": {"quantity": 7, "cost_basis": 8.56},
    "JOBY": {"quantity": 14, "cost_basis": 15.56},
    "KKR":  {"quantity": 19, "cost_basis": 142.77},
    "SOFI": {"quantity": 5, "cost_basis": 27.07},
    "SPYD": {"quantity": 1, "cost_basis": 44.01},
}

# Sidebar options
st.sidebar.header("Settings")
hist_period = st.sidebar.selectbox("History period for quant metrics", ["6mo", "1y", "2y", "5y"], index=1)
perf_period = st.sidebar.selectbox("Performance chart period", ["6mo", "1y", "2y", "5y"], index=1)
sma_window = st.sidebar.number_input("SMA window (days)", min_value=10, max_value=200, value=50, step=10)
mom_lookback = st.sidebar.selectbox("Momentum lookback", ["3mo", "6mo", "1y"], index=2)
st.sidebar.markdown(" ")
refresh = st.sidebar.button("Refresh data")

# ==========================
# 2) HELPERS & CACHING
# ==========================
@st.cache_data(ttl=300)
def fetch_live_price(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            return np.nan
        return float(hist["Close"].iloc[-1])
    except Exception:
        return np.nan

@st.cache_data(ttl=300)
def fetch_history(tickers, period):
    try:
        data = yf.download(tickers, period=period)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=(tickers if isinstance(tickers, str) else data.name))
        return data.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def annualize_return(returns, periods_per_year=252):
    return returns.mean() * periods_per_year

def annualize_vol(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def cagr_from_series(series):
    series = series.dropna()
    if series.empty:
        return np.nan
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    days = (series.index[-1] - series.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    if years and years > 0:
        return (end_val / start_val) ** (1 / years) - 1
    return np.nan

def max_drawdown(series):
    series = series.dropna()
    if series.empty:
        return np.nan
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min()

# ==========================
# 3) FETCH PRICES & BUILD POSITIONS
# ==========================
tickers = list(portfolio.keys())

prices = {t: fetch_live_price(t) for t in tickers}
rows = []
for t, info in portfolio.items():
    qty = info["quantity"]
    cost = info["cost_basis"]
    price = prices.get(t, np.nan)
    market_value = qty * price if not pd.isna(price) else np.nan
    cost_value = qty * cost
    unrealized_pl = market_value - cost_value if not pd.isna(market_value) else np.nan
    return_pct = (unrealized_pl / cost_value) * 100 if cost_value != 0 else np.nan

    rows.append({
        "Ticker": t,
        "Quantity": qty,
        "Price": round(price, 2) if not pd.isna(price),
        "Market Value": round(market_value, 2) if not pd.isna(market_value) else None,
        "Cost Basis": cost,
        "Total Cost": round(cost_value, 2),
        "Unrealized P/L": round(unrealized_pl, 2) if not pd.isna(unrealized_pl) else None,
        "Return %": round(return_pct, 2) if not pd.isna(return_pct) else None,
    })

df = pd.DataFrame(rows)
# avoid divide-by-zero if market values are NaN
if df["Market Value"].sum() == 0 or np.isnan(df["Market Value"].sum()):
    df["Allocation %"] = 0
else:
    df["Allocation %"] = (df["Market Value"] / df["Market Value"].sum()) * 100

total_value = df["Market Value"].sum()
total_cost = df["Total Cost"].sum()
total_pl = total_value - total_cost
total_return = (total_pl / total_cost) * 100 if total_cost != 0 else np.nan

# ==========================
# 4) QUANT METRICS
# ==========================
hist = fetch_history(tickers, hist_period)
spy_hist = fetch_history("SPY", hist_period)
spy_ret = spy_hist["SPY"].pct_change().dropna() if "SPY" in spy_hist.columns else pd.Series(dtype=float)

vol_list = []
sharpe_list = []
momentum_list = []
zscore_list = []
beta_list = []
alpha_list = []

for t in tickers:
    if t not in hist.columns or hist[t].dropna().empty:
        vol_list.append(np.nan); sharpe_list.append(np.nan); momentum_list.append(np.nan)
        zscore_list.append(np.nan); beta_list.append(np.nan); alpha_list.append(np.nan)
        continue

    prices_series = hist[t].dropna()
    returns = prices_series.pct_change().dropna()

    vol = annualize_vol(returns)
    vol_list.append(round(vol, 4))

    sr = (annualize_return(returns) / (returns.std() * np.sqrt(252))) if returns.std() != 0 else np.nan
    sharpe_list.append(round(sr, 4) if not pd.isna(sr) else np.nan)

    if mom_lookback == "3mo": lb_days = 63
    elif mom_lookback == "6mo": lb_days = 126
    else: lb_days = 252
   
    if len(prices_series) > lb_days:
        mom = prices_series.iloc[-1] / prices_series.iloc[-lb_days] - 1
    else:
        mom = prices_series.iloc[-1] / prices_series.iloc[0] - 1
    momentum_list.append(round(mom, 4))

    if len(prices_series) >= sma_window:
        rolling_mean = prices_series.rolling(sma_window).mean().iloc[-1]
        rolling_std = prices_series.rolling(sma_window).std().iloc[-1]
        z = (prices_series.iloc[-1] - rolling_mean) / rolling_std if rolling_std and not pd.isna(rolling_std) else np.nan
    else:
        z = np.nan
    zscore_list.append(round(z, 4) if not pd.isna(z) else np.nan)

    if not spy_ret.empty:
        asset_ret = returns.reindex(spy_ret.index).dropna()
        spy_aligned = spy_ret.reindex(asset_ret.index).dropna()
        asset_ret = asset_ret.reindex(spy_aligned.index)
        if len(asset_ret) > 2:
            cov = np.cov(asset_ret, spy_aligned)[0, 1]
            var_spy = np.var(spy_aligned)
            beta = cov / var_spy if var_spy != 0 else np.nan
            alpha = annualize_return(asset_ret) - (beta * annualize_return(spy_aligned))
        else:
            beta, alpha = np.nan, np.nan
    else:
        beta, alpha = np.nan, np.nan

    beta_list.append(round(beta, 4) if not pd.isna(beta) else np.nan)
    alpha_list.append(round(alpha, 4) if not pd.isna(alpha) else np.nan)


df["Volatility (ann)"] = vol_list
df["Sharpe (ann)"] = sharpe_list
df["Momentum"] = momentum_list
df["Z-Score"] = zscore_list
df["Beta"] = beta_list
df["Alpha (ann)"] = alpha_list

# ==========================
# 5) RISK MODELS (patched risk-parity)
# ==========================
# Build robust risk parity weights using vol estimates from df where possible
vols = df["Volatility (ann)"].replace([0, np.inf, -np.inf], np.nan)
if vols.isna().all():
    # fallback to equal weight when vols are not available
    risk_parity_weights = pd.Series(1 / len(df), index=df["Ticker"]) if len(df) > 0 else pd.Series(dtype=float)
else:
    # fill NaNs with the mean of available vols to avoid zeros
    vols = vols.fillna(vols.mean())
    # ensure no non-positive values
    vols = vols.clip(lower=1e-8)
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    risk_parity_weights = weights.fillna(0)

# store display percent
if isinstance(risk_parity_weights, pd.Series):
    df["Risk Parity Weight %"] = (risk_parity_weights.reindex(df["Ticker"]).fillna(0) * 100).round(2)
else:
    df["Risk Parity Weight %"] = np.nan

ret_hist = hist.pct_change().dropna()
ret_hist = ret_hist[[c for c in ret_hist.columns if c in tickers]].dropna(axis=1, how='all').dropna(axis=0, how='all')

if not ret_hist.empty and len(ret_hist.columns) > 0 and ret_hist.shape[0] > 20:
    mu = ret_hist.mean() * 252
    cov = ret_hist.cov() * 252
    cov += np.eye(len(cov)) * 1e-8
    try:
        inv_cov = np.linalg.inv(cov.values)
        w = inv_cov.dot(mu.values)
        w = np.maximum(w, 0)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        markowitz_weights = pd.Series(index=mu.index, data=w)
        df["Markowitz Weight %"] = df["Ticker"].map(lambda t: round(markowitz_weights.get(t, 0) * 100, 2))
    except:
        df["Markowitz Weight %"] = np.nan
else:
    df["Markowitz Weight %"] = np.nan

# ==========================
# 6) PERFORMANCE TIME SERIES
# ==========================
perf_hist = fetch_history(tickers, perf_period)
if perf_hist.empty:
    perf_hist = fetch_history(tickers, "1y")
if isinstance(perf_hist, pd.Series):
    perf_hist = perf_hist.to_frame(name=perf_hist.name)
available_cols = [c for c in perf_hist.columns if c in tickers]
perf_hist = perf_hist[available_cols].dropna(how="all")

shares = pd.Series({t: portfolio[t]["quantity"] for t in tickers})
shares = shares.reindex(perf_hist.columns).fillna(0)
daily_values = perf_hist.mul(shares, axis=1)
portfolio_value = daily_values.sum(axis=1)

spy_perf = fetch_history("SPY", perf_period)
spy_series = spy_perf["SPY"] if "SPY" in spy_perf.columns else spy_perf.squeeze() if not spy_perf.empty else pd.Series(dtype=float)

def normalize_series(s):
    s = s.dropna()
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

portfolio_norm = normalize_series(portfolio_value)
spy_norm = normalize_series(spy_series)

# ==========================
# 7) BACKTESTING
# ==========================

def rebalance_instructions(current_shares, current_prices, target_weights, total_portfolio_value):
    target_values = target_weights * total_portfolio_value
    target_shares = (target_values / current_prices).fillna(0).apply(np.floor)
    trades = target_shares.subtract(current_shares.reindex(target_shares.index).fillna(0)).astype(int)
    cash_used = (target_shares * current_prices).sum()
    cash_left = total_portfolio_value - cash_used
    return target_shares.astype(int), trades.astype(int), cash_left

def backtest_monthly_rebalance(price_df, initial_weights, rebalance_weights_func, start_cash=0):
    price_df = price_df.dropna(how="all").fillna(method="ffill").dropna(axis=1, how="all")
    if price_df.empty:
        return pd.Series(dtype=float)

    dates = price_df.index
    tickers_bt = price_df.columns.tolist()

    capital = total_value if not pd.isna(total_value) and total_value > 0 else 10000.0

    init_weights = initial_weights.reindex(tickers_bt).fillna(0)
    target_values = init_weights * capital
    shares_bt = (target_values / price_df.iloc[0]).fillna(0).apply(np.floor).astype(int)
    cash = capital - (shares_bt * price_df.iloc[0]).sum()

    equity = []
    current_shares = shares_bt.copy()

    rebalance_dates = sorted({d.replace(day=1) for d in dates})

    actual_reb_dates = []
    for r in rebalance_dates:
        candidates = dates[dates >= r]
        if not candidates.empty:
            actual_reb_dates.append(candidates[0])

    for dt in dates:
        if dt in actual_reb_dates:
            new_weights = rebalance_weights_func(dt, price_df.loc[:dt])
            new_weights = new_weights.reindex(tickers_bt).fillna(0)
            target_vals = new_weights * ((current_shares * price_df.loc[dt]).sum() + cash)
            target_shares = (target_vals / price_df.loc[dt]).fillna(0).apply(np.floor).astype(int)
            cash += (current_shares - target_shares) @ price_df.loc[dt]
            current_shares = target_shares

        daily_value = (current_shares * price_df.loc[dt]).sum() + cash
        equity.append(daily_value)

    return pd.Series(index=dates, data=equity)

def compute_metrics_from_equity(eq_series):
    if eq_series.empty:
        return {}
    cum_return = eq_series.iloc[-1] / eq_series.iloc[0] - 1
    cagr = cagr_from_series(eq_series)
    daily_ret = eq_series.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252) if not daily_ret.empty else np.nan
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else np.nan
    mdd = max_drawdown(eq_series)
    return {"Cumulative Return": cum_return, "CAGR": cagr, "Ann Vol": ann_vol, "Sharpe": sharpe, "Max Drawdown": mdd}

# ==========================
# 8) STREAMLIT UI (TABS)
# ==========================

tab_overview, tab_quant, tab_risk, tab_perf, tab_signals = st.tabs([
    "Overview", "Quant Metrics", "Risk Models", "Performance", "Signals"
])

with tab_overview:
    st.subheader("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"${total_value:,.2f}")
    col2.metric("Total Cost", f"${total_cost:,.2f}")
    col3.metric("Unrealized P/L", f"${total_pl:,.2f}", f"{total_return:.2f}%")
    col4.metric("# Positions", len(df))

    st.markdown("**Holdings**")
    st.dataframe(df.sort_values("Market Value", ascending=False).reset_index(drop=True), use_container_width=True)

with tab_quant:
    st.subheader("Per-Asset Quant Metrics")
    show_cols = ["Ticker", "Price", "Market Value", "Allocation %", "Volatility (ann)", "Sharpe (ann)",
                 "Momentum", "Z-Score", "Beta", "Alpha (ann)"]
    st.dataframe(df[show_cols].sort_values("Market Value", ascending=False).reset_index(drop=True), use_container_width=True)

    if not ret_hist.empty:
        corr = ret_hist.corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Return Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_risk:
    st.subheader("Risk Models & Rebalance")
    st.dataframe(df[["Ticker", "Market Value", "Allocation %", "Risk Parity Weight %", "Markowitz Weight %"]]
                 .sort_values("Market Value", ascending=False).reset_index(drop=True), use_container_width=True)

    st.markdown("### Rebalance Panel")
    colA, colB = st.columns([2, 3])

    with colA:
        strategy_choice = st.selectbox("Select target allocation strategy",
                                       ["Current Allocation", "Equal Weight", "Risk Parity", "Markowitz", "Momentum Tilt"],
                                       index=0)
        invest_amount = st.number_input("Amount to invest / rebalance (USD)", value=float(total_value if not pd.isna(total_value) else 10000.0))
        apply_btn = st.button("Compute Rebalance Instructions")

    with colB:
    
        # -------------------------
        # TARGET WEIGHTS CALCULATION
        # -------------------------
        def compute_risk_parity_weights(price_df):
            """Compute robust inverse-volatility (risk parity) weights"""
            returns = price_df.pct_change().dropna()
            if returns.shape[0] < 20 or returns.empty:
                return pd.Series(1/len(price_df.columns), index=price_df.columns)
        
            vol = returns.std() * np.sqrt(252)
            vol = vol.fillna(vol.mean())
            vol = vol.clip(lower=1e-8)
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            return weights

    # -------------------------
    # STRATEGY SELECTION
    # -------------------------
    if strategy_choice == "Current Allocation":
        target_weights = (df["Market Value"] / df["Market Value"].sum()).fillna(0)
    elif strategy_choice == "Equal Weight":
        target_weights = pd.Series(1/len(df), index=df["Ticker"])
    elif strategy_choice == "Risk Parity":
        hist_for_rp = hist[df["Ticker"]].dropna(how='all')
        if hist_for_rp.empty:
            target_weights = pd.Series(1/len(df), index=df["Ticker"])
        else:
            target_weights = compute_risk_parity_weights(hist_for_rp)
            target_weights = target_weights.reindex(df["Ticker"]).fillna(0)
    elif strategy_choice == "Markowitz":
        target_weights = df.set_index("Ticker")["Markowitz Weight %"].fillna(0) / 100
        if target_weights.sum() == 0:
            target_weights = pd.Series(1/len(df), index=df["Ticker"])
    else:  # Momentum Tilt
        m = df.set_index("Ticker")["Momentum"].replace([np.inf, -np.inf], np.nan).fillna(-999)
        n_top = max(1, int(len(m) * 0.5))
        top_tickers = m.sort_values(ascending=False).iloc[:n_top].index
        w = pd.Series(0, index=df["Ticker"])
        w[top_tickers] = 1/len(top_tickers)
        target_weights = w

    # Reindex to match df tickers and fill missing
    target_weights = target_weights.reindex(df["Ticker"]).fillna(0)

    st.dataframe((target_weights * 100).round(2).rename("Target %").to_frame(), use_container_width=True)


    if apply_btn:
        current_shares = pd.Series({t: portfolio[t]["quantity"] for t in df["Ticker"]})
        current_prices = pd.Series({t: prices.get(t, np.nan) for t in df["Ticker"]})
        target_weights = target_weights.reindex(df["Ticker"]).fillna(0)

        target_shares, trades, cash_left = rebalance_instructions(current_shares, current_prices, target_weights, invest_amount)

        instr_df = pd.DataFrame({
            "Ticker": target_shares.index,
            "Current Shares": current_shares,
            "Target Shares": target_shares,
            "Trade (buy + / sell -)": trades,
            "Price": current_prices.round(2),
            "Trade $": (trades * current_prices).round(2)
        }).set_index("Ticker")

        st.dataframe(instr_df, use_container_width=True)
        st.markdown(f"Cash leftover: **${cash_left:,.2f}**")

with tab_perf:
    st.subheader("Portfolio Performance")
    if not portfolio_value.empty:
        fig_val = px.line(x=portfolio_value.index, y=portfolio_value.values, title="Portfolio Value")
        st.plotly_chart(fig_val, use_container_width=True)

    st.markdown("---")
    st.subheader("Backtesting (monthly rebalance)")

    backtest_strategy = st.selectbox("Choose strategy to backtest",
                                     ["Buy & Hold (current)", "Monthly Risk-Parity", "Monthly Momentum Tilt (top 50%)", "Monthly Markowitz"], index=0)

    bt_start = st.date_input("Backtest start date", value=(datetime.today() - pd.DateOffset(years=1)).date())
    bt_end = st.date_input("Backtest end date", value=datetime.today().date())
    bt_run = st.button("Run Backtest")

    if bt_run:
        price_df_full = fetch_history(tickers, "5y")
        price_df_full = price_df_full.loc[(price_df_full.index.date >= bt_start) & (price_df_full.index.date <= bt_end)]

        def rp_weights_func(dt, subset):
            # robust risk parity weight calculation using covariance diagonal
            reth = subset.pct_change().dropna()
            if reth.shape[0] < 20:
                return pd.Series(1 / len(subset.columns), index=subset.columns)

            cov = reth.cov()
            vol = np.sqrt(np.diag(cov))
            vol = pd.Series(vol, index=subset.columns)
            vol = vol.replace([np.inf, -np.inf], np.nan)
            if vol.isna().any():
                vol = vol.fillna(vol.mean())
            # guard against non-positive
            vol = vol.clip(lower=1e-8)
            if vol.sum() == 0:
                return pd.Series(1 / len(subset.columns), index=subset.columns)
            invv = 1 / vol
            w = invv / invv.sum()
            return w

        def momentum_weights_func(dt, subset):
            lookback = 126 if mom_lookback == "6mo" else (63 if mom_lookback == "3mo" else 252)
            scores = {}
            for col in subset.columns:
                s = subset[col].dropna()
                if len(s) > lookback:
                    scores[col] = s.iloc[-1] / s.iloc[-lookback] - 1
                elif len(s) > 1:
                    scores[col] = s.iloc[-1] / s.iloc[0] - 1
                else:
                    scores[col] = -999
            mom_s = pd.Series(scores)
            n_top = max(1, int(len(mom_s) * 0.5))
            top = mom_s.sort_values(ascending=False).iloc[:n_top].index
            w = pd.Series(0, index=mom_s.index)
            if len(top) > 0:
                w[top] = 1 / len(top)
            return w

        def markowitz_weights_func(dt, subset):
            reth = subset.pct_change().dropna()
            if reth.shape[0] < 20:
                return pd.Series(1/len(subset.columns), index=subset.columns)
            mu = reth.mean() * 252
            cov = reth.cov() * 252
            cov += np.eye(len(cov)) * 1e-8
            try:
                inv_cov = np.linalg.inv(cov.values)
                w = inv_cov.dot(mu.values)
                w = np.maximum(w, 0)
                if w.sum() == 0:
                    w = np.ones_like(w) / len(w)
                else:
                    w = w / w.sum()
                return pd.Series(w, index=mu.index)
            except:
                return pd.Series(1/len(subset.columns), index=subset.columns)

        if backtest_strategy == "Buy & Hold (current)":
            init_w = (df.set_index("Ticker")["Market Value"] / df["Market Value"].sum()).reindex(tickers).fillna(0)
            strat_func = lambda dt, sub: init_w
        elif backtest_strategy == "Monthly Risk-Parity":
            init_w = pd.Series(1/len(tickers), index=tickers)
            strat_func = rp_weights_func
        elif backtest_strategy == "Monthly Momentum Tilt (top 50%)":
            init_w = pd.Series(1/len(tickers), index=tickers)
            strat_func = momentum_weights_func
        else:
            init_w = pd.Series(1/len(tickers), index=tickers)
            strat_func = markowitz_weights_func

        eq = backtest_monthly_rebalance(price_df_full, init_w, strat_func)
        if not eq.empty:
            fig_bt = px.line(x=eq.index, y=eq.values, title=f"Equity Curve — {backtest_strategy}")
            st.plotly_chart(fig_bt, use_container_width=True)

            metr = compute_metrics_from_equity(eq)
            metr_df = pd.DataFrame(metr, index=[0]).T
            metr_df.columns = ["Value"]
            st.dataframe(metr_df, use_container_width=True)

with tab_signals:
    st.subheader("Trading Signals & Alerts")
    
    st.markdown("**Debug: per-ticker history lengths & NaN%**")
    lengths = {t: int(hist[t].dropna().shape[0]) if t in hist.columns else 0 for t in tickers}
    nan_pct = {t: float(hist[t].isna().mean()) if t in hist.columns else 1.0 for t in tickers}
    debug_df = pd.DataFrame({"rows": lengths, "nan_pct": nan_pct})
    st.write(debug_df)


    # =========================================================
    # 1) SMA CROSSOVER SIGNALS
    # =========================================================
    sma_signal = {}
    for t in tickers:
        series = hist[t].dropna()
        if len(series) >= sma_window:
            sma = series.rolling(sma_window).mean()
            sma_signal[t] = "Buy" if series.iloc[-1] > sma.iloc[-1] else "Sell"
        else:
            sma_signal[t] = "N/A"
    sma_df = pd.DataFrame.from_dict(sma_signal, orient="index", columns=["SMA Signal"])

    st.markdown("### SMA Signals")
    st.dataframe(
        sma_df.style.applymap(
            lambda v: "color: white; background-color: green" if v == "Buy" else
                      "color: white; background-color: red" if v == "Sell" else
                      "color: black; background-color: lightgray"
        ),
        use_container_width=True
    )

    # =========================================================
    # 2) MOMENTUM SIGNALS
    # =========================================================
    mom_signal = {}
    lb_days = 126 if mom_lookback == "6mo" else (63 if mom_lookback == "3mo" else 252)

    for t in tickers:
        series = hist[t].dropna()
        if len(series) > lb_days:
            mom = series.iloc[-1] / series.iloc[-lb_days] - 1
            mom_signal[t] = "Buy" if mom > 0 else "Sell"
        else:
            mom_signal[t] = "N/A"

    mom_df = pd.DataFrame.from_dict(mom_signal, orient="index", columns=["Momentum Signal"])

    st.markdown("### Momentum Signals")
    st.dataframe(
        mom_df.style.applymap(
            lambda v: "color: white; background-color: green" if v == "Buy" else
                      "color: white; background-color: red" if v == "Sell" else
                      "color: black; background-color: lightgray"
        ),
        use_container_width=True
    )

    # =========================================================
    # 3) RISK PARITY SIGNALS
    # =========================================================
    def compute_risk_parity_weights(price_df):
        returns = price_df.pct_change().dropna()
        if returns.shape[0] < 20:
            return pd.Series(1/len(price_df.columns), index=price_df.columns)

        vol = returns.std() * np.sqrt(252)
        vol = vol.replace(0, np.nan).fillna(vol.mean())
        inv_vol = 1 / vol
        return inv_vol / inv_vol.sum()

    hist_for_rp = hist[df["Ticker"]].dropna(how="all")
    rp_weights = compute_risk_parity_weights(hist_for_rp)

    rp_signal = {}
    for t in df["Ticker"]:
        eq_w = 1 / len(df)
        rp_signal[t] = "Overweight" if rp_weights.get(t, 0) > eq_w else "Underweight"

    rp_df = pd.DataFrame.from_dict(rp_signal, orient="index", columns=["Risk Parity Signal"])

    st.markdown("### Risk Parity Signals")
    st.dataframe(
        rp_df.style.applymap(
            lambda v: "color: white; background-color: green" if v == "Overweight" else
                      "color: white; background-color: red" if v == "Underweight" else
                      "color: black; background-color: lightgray"
        ),
        use_container_width=True
    )

        # =========================================================
    # 7) 52-WEEK HIGH BREAKOUT SIGNAL
    # =========================================================
    breakout_signal = {}
    for t in tickers:
        series = hist[t].dropna()
        if len(series) < 252:
            breakout_signal[t] = "N/A"
            continue
        
        high_52w = series[-252:].max()
        last = series.iloc[-1]
        pct_from_high = (last / high_52w) - 1

        if pct_from_high >= -0.03:  # within 3%
            breakout_signal[t] = "Breakout (Buy)"
        elif pct_from_high <= -0.10:
            breakout_signal[t] = "Deep Pullback"
        else:
            breakout_signal[t] = "Neutral"

    breakout_df = pd.DataFrame.from_dict(breakout_signal, orient="index",
                                         columns=["52W High Signal"])

    def breakout_color(v):
        if "Breakout" in v:
            return "color: white; background-color: green"
        if "Pullback" in v:
            return "color: white; background-color: blue"
        if v == "Neutral":
            return "background-color: lightgray"
        return ""

    st.markdown("### 52-Week High Breakout Signals")
    st.dataframe(breakout_df.style.applymap(breakout_color), use_container_width=True)


    # =========================================================
    # 8) VOLATILITY REGIME SIGNAL
    # =========================================================
    vol_regime_signal = {}
    for t in tickers:
        series = hist[t].dropna()
        if len(series) < 100:
            vol_regime_signal[t] = "N/A"
            continue

        # Daily returns volatility
        ret = series.pct_change().dropna()
        current_vol = ret.rolling(20).std().iloc[-1]  # 20-day vol
        median_vol = ret.rolling(20).std().median()

        if current_vol > 1.25 * median_vol:
            vol_regime_signal[t] = "High Vol (Risk Off)"
        elif current_vol < 0.75 * median_vol:
            vol_regime_signal[t] = "Low Vol (Risk On)"
        else:
            vol_regime_signal[t] = "Neutral"

    vol_df = pd.DataFrame.from_dict(vol_regime_signal, orient="index",
                                    columns=["Volatility Regime"])

    def vol_color(v):
        if "High Vol" in v:
            return "color: white; background-color: red"
        if "Low Vol" in v:
            return "color: white; background-color: green"
        if "Neutral" in v:
            return "background-color: lightgray"
        return ""

    st.markdown("### Volatility Regime Signals")
    st.dataframe(vol_df.style.applymap(vol_color), use_container_width=True)


    # =========================================================
    # 4) COMBINED SIGNAL TABLE (FULL COLOR)
    # =========================================================
    combined_df = sma_df.join(mom_df).join(rp_df)
    combined_df = combined_df.join(breakout_df).join(vol_df)

    def color_map(val):
        if val in ("Buy", "Overweight"):
            return "color: white; background-color: green"
        elif val in ("Sell", "Underweight"):
            return "color: white; background-color: red"
        else:
            return "color: black; background-color: lightgray"

    st.markdown("### Combined Signals")
    st.dataframe(
        combined_df.style.applymap(color_map),
        use_container_width=True
    )

    
