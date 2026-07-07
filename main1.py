import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st

# Data Loader
def load_data(tickers, start='2020-01-01', end='2024-12-31'):
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        df = data['Close']
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        df = data['Close']
    else:
        raise ValueError("No 'Close' data found in the downloaded dataset.")
    df = df.ffill().dropna(axis=0, how='any')
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        print(f"Warning: Could not download data for: {missing}")
    return df

# Beta Calculation
def calculate_betas(returns, market_column):
    market_returns = returns[market_column]
    return pd.DataFrame({
        stock: returns[stock].cov(market_returns) / market_returns.var()
        for stock in returns.columns if stock != market_column
    }, index=['Beta']).T

# Risk Metrics
#variance
def calculate_var(returns, confidence=0.95):
    mean = returns.mean()
    std = returns.std()
    return norm.ppf(1 - confidence, mean, std)

#Jensens_alpha
def calculate_jensens_alpha(portfolio_return, beta, market_return, risk_free_rate):
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return portfolio_return - expected_return

#sortino_ratio
def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    downside_returns = returns[returns < risk_free_rate]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.nan
    return (returns.mean() - risk_free_rate) / downside_std

#max_drawdown
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

#variance_histogram
def plot_var_histogram(returns, var_95, var_99, save_path=None):
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(returns, bins=50, kde=True, color='skyblue', ax=ax)
    ax.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.2%}')
    ax.axvline(var_99, color='darkred', linestyle='--', label=f'VaR 99%: {var_99:.2%}')
    ax.set_title('Portfolio Return Distribution with VaR')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300)
    else:
        st.pyplot(fig)

    plt.close(fig)  # Always close the figure

# Portfolio Simulation
def simulate_portfolios(returns, num_portfolios=5000, risk_free_rate=0.02):
    np.random.seed(42)
    num_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results = []
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        results.append([ret, vol, sharpe, weights])
    return pd.DataFrame(results, columns=['Returns', 'Volatility', 'Sharpe Ratio', 'Weights']).dropna()

# Portfolio Optimization
def optimize_portfolio(returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(returns.columns)

    def performance(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return ret, vol, sharpe

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    init_guess = num_assets * [1. / num_assets]

    max_sharpe = minimize(lambda w: -performance(w)[2], init_guess, bounds=bounds, constraints=constraints)
    min_vol = minimize(lambda w: performance(w)[1], init_guess, bounds=bounds, constraints=constraints)

    return {
        'Max Sharpe': {'Weights': max_sharpe.x, 'Performance': performance(max_sharpe.x)},
        'Min Volatility': {'Weights': min_vol.x, 'Performance': performance(min_vol.x)}
    }

# Plot Efficient Frontier
def plot_efficient_frontier(df, save_path=None):
    import streamlit as st
    import matplotlib.pyplot as plt

    if df.empty or df['Sharpe Ratio'].isna().all():
        st.warning("Efficient frontier data is empty or invalid.")
        return

    df = df.dropna(subset=['Sharpe Ratio'])
    max_idx = df['Sharpe Ratio'].idxmax()
    max_sharpe = df.loc[max_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Volatility'], df['Returns'], c=df['Sharpe Ratio'], cmap='viridis', edgecolors='k')
    fig.colorbar(scatter, label='Sharpe Ratio')
    ax.scatter(max_sharpe['Volatility'], max_sharpe['Returns'], color='red', marker='*', s=200, label='Max Sharpe')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        st.pyplot(fig)
        plt.close(fig)  # close after rendering

def generate_heatmaps(returns):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    os.makedirs("outputs", exist_ok=True)

    # Correlation Matrix Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("outputs/correlation_matrix.png", dpi=300)
    plt.close()

    # Covariance Matrix Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(returns.cov(), annot=True, cmap='coolwarm')
    plt.title('Covariance Matrix')
    plt.tight_layout()
    plt.savefig("outputs/covariance_matrix.png", dpi=300)
    plt.close()

# Main Function
def main():
    tickers = ['RELIANCE.NS', 'ICICIBANK.NS', 'TCS.NS', 'INFY.NS', 'ITC.NS']
    market_index = '^NSEI'
    all_tickers = tickers + [market_index]

    data = load_data(all_tickers)
    returns = data.pct_change().dropna()
    os.makedirs('outputs', exist_ok=True)

    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('outputs/correlation_matrix.png', dpi=300)
    plt.close()

    sns.heatmap(returns.cov(), annot=True, cmap='coolwarm')
    plt.title('Covariance Matrix')
    plt.savefig('outputs/covariance_matrix.png', dpi=300)
    plt.close()

    beta_df = calculate_betas(returns, market_index)
    beta_df.to_excel('outputs/beta_values.xlsx')

    portfolio_df = simulate_portfolios(returns[tickers])
    plot_efficient_frontier(portfolio_df, 'outputs/efficient_frontier.png')

    results = optimize_portfolio(returns[tickers])
    market_return = returns[market_index].mean() * 252
    summary_data = []

    for label, res in results.items():
        weights = res['Weights']
        ret, vol, sharpe = res['Performance']
        port_returns = returns[tickers] @ weights
        var_95 = calculate_var(port_returns, 0.95)
        var_99 = calculate_var(port_returns, 0.99)
        sortino = calculate_sortino_ratio(port_returns)
        drawdown = calculate_max_drawdown(port_returns)
        beta = sum(weights[i] * beta_df.loc[tickers[i], 'Beta'] for i in range(len(tickers)))
        alpha = calculate_jensens_alpha(ret, beta, market_return, 0.02)

        plot_var_histogram(port_returns, var_95, var_99, f'outputs/{label}_var_histogram.png')

        print(f"\n{label} Portfolio")
        for t, w in zip(tickers, weights):
            print(f"  {t}: {w:.2%}")
        print(f"  Return: {ret:.2%}, Volatility: {vol:.2%}, Sharpe: {sharpe:.2f}")
        print(f"  VaR 95%: {var_95:.2%}, VaR 99%: {var_99:.2%}, Sortino: {sortino:.2f},Jensen's Alpha: {alpha:.2%}, Max DD: {drawdown:.2%}")

        summary_data.append({
            'Portfolio': label,
            'Expected Return': ret,
            'Volatility': vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Beta': beta,
            'Jensen Alpha': alpha,
            'VaR 95%': var_95,
            'VaR 99%': var_99,
            'Max Drawdown': drawdown
        })

    pd.DataFrame(summary_data).to_excel("outputs/portfolio_summary.xlsx", index=False)
    
if __name__ == "__main__":
    main()
