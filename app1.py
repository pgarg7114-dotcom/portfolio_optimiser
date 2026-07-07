import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

from main1 import (
    load_data, calculate_betas, simulate_portfolios,
    optimize_portfolio, calculate_var, calculate_jensens_alpha,
    calculate_sortino_ratio, calculate_max_drawdown,
    plot_efficient_frontier, plot_var_histogram,
    generate_heatmaps  # heatmap generator
)

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization & Risk Analysis")

# Sidebar Inputs
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.multiselect(
    "Choose Stocks", options=[
        'RELIANCE.NS', 'ICICIBANK.NS', 'TCS.NS', 'INFY.NS', 'ITC.NS',
        'HDFCBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'DRREDDY.NS',
        'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'
    ],
    default=['RELIANCE.NS', 'ICICIBANK.NS', 'TCS.NS']
)

risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

if st.sidebar.button("Run Optimization") and tickers:
    with st.spinner("⏳ Running portfolio analysis..."):
        market_index = '^NSEI'
        all_tickers = tickers + [market_index]

        # Load & process data
        data = load_data(all_tickers, start=start_date, end=end_date)
        returns = data.pct_change().dropna()

        # Generate and display heatmaps
        generate_heatmaps(returns)  # this saves both heatmaps

        st.subheader("Correlation Matrix")
        cor_path = "outputs/correlation_matrix.png"
        if os.path.exists(cor_path):
            st.image(Image.open(cor_path), use_column_width=True)

        st.subheader("Covariance Matrix")
        cov_path = "outputs/covariance_matrix.png"
        if os.path.exists(cov_path):
            st.image(Image.open(cov_path), use_column_width=True)

        # Run simulation & optimization
        beta_df = calculate_betas(returns, market_index)
        portfolio_df = simulate_portfolios(returns[tickers])
        results = optimize_portfolio(returns[tickers], risk_free_rate)
        market_return = returns[market_index].mean() * 252

        # Efficient Frontier
        st.subheader("Efficient Frontier")
        plot_efficient_frontier(portfolio_df)  # no double render

        for label, res in results.items():
            weights = res['Weights']
            ret, vol, sharpe = res['Performance']
            port_returns = returns[tickers] @ weights

            var_95 = calculate_var(port_returns, 0.95)
            var_99 = calculate_var(port_returns, 0.99)
            sortino = calculate_sortino_ratio(port_returns, risk_free_rate)
            drawdown = calculate_max_drawdown(port_returns)
            beta = sum(weights[i] * beta_df.loc[tickers[i], 'Beta'] for i in range(len(tickers)))
            alpha = calculate_jensens_alpha(ret, beta, market_return, risk_free_rate)

            st.markdown(f"### 📌 {label} Portfolio")
            st.write(pd.DataFrame({"Weight": weights}, index=tickers).style.format("{:.2%}"))

            st.metric("Expected Return", f"{ret:.2%}")
            st.metric("Volatility", f"{vol:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Sortino Ratio", f"{sortino:.2f}")
            st.metric("Jensen's Alpha", f"{alpha:.2%}")
            st.metric("Max Drawdown", f"{drawdown:.2%}")
            st.metric("VaR (95%)", f"{var_95:.2%}")
            st.metric("VaR (99%)", f"{var_99:.2%}")

            # Plot VaR Histogram
            st.subheader("Value-at-Risk Histogram")
            plot_var_histogram(port_returns, var_95, var_99, save_path=None)  
else:
    st.info("💡 Select at least one stock and click 'Run Optimization' to begin.")
