"""
Stock Market Analysis System
Author: Vishvaa (vishvaa369@gmail.com)
GitHub: https://github.com/vishvaa369

A comprehensive Python-based stock market analysis system that evaluates
historical stock price data, identifies market trends, and generates
insightful graphical reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "monospace",
})

PALETTE = {
    "AAPL": "#58a6ff",
    "GOOGL": "#3fb950",
    "MSFT": "#f78166",
    "AMZN": "#d2a8ff",
    "TSLA": "#ffa657",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_stock_data(ticker: str, start: str = "2022-01-01",
                        end: str = "2024-12-31",
                        seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV stock data using geometric Brownian motion.

    Parameters
    ----------
    ticker : str   Stock symbol (e.g. 'AAPL')
    start  : str   Start date  'YYYY-MM-DD'
    end    : str   End date    'YYYY-MM-DD'
    seed   : int   Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume, Ticker
    """
    params = {
        "AAPL":  {"price": 170, "mu": 0.0003, "sigma": 0.018, "seed": seed},
        "GOOGL": {"price": 140, "mu": 0.0002, "sigma": 0.020, "seed": seed + 1},
        "MSFT":  {"price": 310, "mu": 0.0004, "sigma": 0.016, "seed": seed + 2},
        "AMZN":  {"price": 185, "mu": 0.0002, "sigma": 0.022, "seed": seed + 3},
        "TSLA":  {"price": 250, "mu": 0.0001, "sigma": 0.035, "seed": seed + 4},
    }

    p = params.get(ticker, {"price": 100, "mu": 0.0002, "sigma": 0.02, "seed": seed})
    np.random.seed(p["seed"])

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    returns = np.random.normal(p["mu"], p["sigma"], n)
    close = p["price"] * np.cumprod(1 + returns)

    daily_range = np.abs(np.random.normal(0, p["sigma"] * 0.6, n)) * close
    open_  = close - np.random.uniform(-0.5, 0.5, n) * daily_range
    high   = np.maximum(close, open_) + np.abs(np.random.normal(0, 0.3, n)) * daily_range
    low    = np.minimum(close, open_) - np.abs(np.random.normal(0, 0.3, n)) * daily_range
    volume = np.random.randint(20_000_000, 120_000_000, n).astype(float)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   np.round(open_,  2),
        "High":   np.round(high,   2),
        "Low":    np.round(low,    2),
        "Close":  np.round(close,  2),
        "Volume": volume,
        "Ticker": ticker,
    })
    df.set_index("Date", inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate stock data.
      - Remove duplicates
      - Forward-fill missing values
      - Remove rows with non-positive prices or volume
      - Validate OHLC consistency
    """
    original_len = len(df)
    df = df[~df.index.duplicated(keep="first")]
    df = df.ffill()
    df = df[(df["Close"] > 0) & (df["Open"] > 0) &
            (df["High"] > 0) & (df["Low"] > 0) & (df["Volume"] > 0)]
    df = df[df["High"] >= df["Low"]]
    removed = original_len - len(df)
    print(f"  [{df['Ticker'].iloc[0]}] Cleaned {removed} bad rows  →  {len(df)} rows remain")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators:
      Moving Averages (20, 50, 200-day)
      Bollinger Bands (20-day, 2σ)
      Daily Return & Cumulative Return
      RSI (14-day)
      MACD (12/26/9)
      ATR (14-day)
      OBV
    """
    # Moving Averages
    for w in (20, 50, 200):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    # Bollinger Bands
    df["BB_mid"]   = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    # Returns
    df["Daily_Return"]      = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    # RSI
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12        = df["Close"].ewm(span=12, adjust=False).mean()
    ema26        = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STATISTICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_statistics(df: pd.DataFrame) -> dict:
    """Return a dict of key performance statistics for a single stock."""
    dr   = df["Daily_Return"].dropna()
    stat = {
        "Ticker":              df["Ticker"].iloc[0],
        "Start_Price":         round(df["Close"].iloc[0],  2),
        "End_Price":           round(df["Close"].iloc[-1], 2),
        "Total_Return_%":      round(df["Cumulative_Return"].iloc[-1] * 100, 2),
        "Annualised_Return_%": round(((df["Close"].iloc[-1] / df["Close"].iloc[0])
                                       ** (252 / len(df)) - 1) * 100, 2),
        "Daily_Volatility_%":  round(dr.std() * 100, 4),
        "Annual_Volatility_%": round(dr.std() * np.sqrt(252) * 100, 2),
        "Sharpe_Ratio":        round(dr.mean() / dr.std() * np.sqrt(252), 2),
        "Max_Drawdown_%":      round(((df["Close"] / df["Close"].cummax()) - 1).min() * 100, 2),
        "Best_Day_%":          round(dr.max() * 100, 2),
        "Worst_Day_%":         round(dr.min() * 100, 2),
        "Avg_Volume":          int(df["Volume"].mean()),
    }
    return stat


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_price_with_ma(df: pd.DataFrame):
    """Closing price with Moving Averages + Bollinger Bands."""
    ticker = df["Ticker"].iloc[0]
    color  = PALETTE.get(ticker, "#58a6ff")

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{ticker}  —  Price & Moving Averages", fontsize=16,
                 color="#e6edf3", fontweight="bold", y=0.98)

    ax = axes[0]
    ax.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                    alpha=0.08, color=color, label="Bollinger Band")
    ax.plot(df.index, df["Close"],  color=color,    lw=1.4, label="Close")
    ax.plot(df.index, df["MA20"],   color="#ffa657", lw=1.2, ls="--", label="MA 20")
    ax.plot(df.index, df["MA50"],   color="#3fb950", lw=1.2, ls="--", label="MA 50")
    ax.plot(df.index, df["MA200"],  color="#d2a8ff", lw=1.4, ls=":",  label="MA 200")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Volume
    ax2 = axes[1]
    ax2.bar(df.index, df["Volume"] / 1e6, color=color, alpha=0.5, width=1)
    ax2.set_ylabel("Volume (M)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_price_ma.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_rsi_macd(df: pd.DataFrame):
    """RSI and MACD panel."""
    ticker = df["Ticker"].iloc[0]
    color  = PALETTE.get(ticker, "#58a6ff")

    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.suptitle(f"{ticker}  —  RSI & MACD", fontsize=16,
                 color="#e6edf3", fontweight="bold", y=0.99)

    # Close
    axes[0].plot(df.index, df["Close"], color=color, lw=1.4)
    axes[0].set_ylabel("Close Price")

    # RSI
    ax_rsi = axes[1]
    ax_rsi.plot(df.index, df["RSI"], color="#ffa657", lw=1.2)
    ax_rsi.axhline(70, color="#f78166", ls="--", lw=0.8, label="Overbought (70)")
    ax_rsi.axhline(30, color="#3fb950", ls="--", lw=0.8, label="Oversold (30)")
    ax_rsi.fill_between(df.index, df["RSI"], 70,
                        where=(df["RSI"] >= 70), alpha=0.2, color="#f78166")
    ax_rsi.fill_between(df.index, df["RSI"], 30,
                        where=(df["RSI"] <= 30), alpha=0.2, color="#3fb950")
    ax_rsi.set_ylabel("RSI (14)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(fontsize=8)

    # MACD
    ax_macd = axes[2]
    ax_macd.plot(df.index, df["MACD"],   color="#58a6ff", lw=1.2, label="MACD")
    ax_macd.plot(df.index, df["Signal"], color="#ffa657", lw=1.2, label="Signal")
    colors = ["#3fb950" if v >= 0 else "#f78166" for v in df["MACD_Hist"]]
    ax_macd.bar(df.index, df["MACD_Hist"], color=colors, alpha=0.6, width=1, label="Histogram")
    ax_macd.axhline(0, color="#8b949e", lw=0.7)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_rsi_macd.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_returns_distribution(df: pd.DataFrame):
    """Daily return histogram + KDE."""
    ticker = df["Ticker"].iloc[0]
    color  = PALETTE.get(ticker, "#58a6ff")
    dr     = df["Daily_Return"].dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker}  —  Return Distribution", fontsize=14,
                 color="#e6edf3", fontweight="bold")

    # Histogram + KDE
    sns.histplot(dr, bins=60, kde=True, color=color, ax=axes[0],
                 edgecolor="none", alpha=0.7)
    axes[0].axvline(dr.mean(), color="#ffa657", ls="--", lw=1.5,
                    label=f"Mean {dr.mean():.2f}%")
    axes[0].axvline(0, color="#8b949e", lw=0.8)
    axes[0].set_xlabel("Daily Return (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Daily Returns")
    axes[0].legend()

    # Q-Q plot approximation with sorted returns vs normal
    sorted_returns = np.sort(dr)
    theoretical    = np.random.normal(dr.mean(), dr.std(), len(sorted_returns))
    theoretical.sort()
    axes[1].scatter(theoretical, sorted_returns, color=color, alpha=0.3, s=5)
    lim = max(abs(sorted_returns.min()), abs(sorted_returns.max()))
    axes[1].plot([-lim, lim], [-lim, lim], color="#ffa657", lw=1.5, ls="--",
                 label="Normal Reference")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")
    axes[1].set_title("Q-Q Plot")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_returns_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_multi_stock_comparison(dfs: list):
    """Normalised cumulative return comparison across all stocks."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Multi-Stock Comparison  —  Cumulative Returns",
                 fontsize=16, color="#e6edf3", fontweight="bold")

    for df in dfs:
        t     = df["Ticker"].iloc[0]
        color = PALETTE.get(t, "#58a6ff")
        cum   = df["Cumulative_Return"] * 100
        axes[0].plot(df.index, cum, color=color, lw=1.6, label=t)

    axes[0].axhline(0, color="#8b949e", lw=0.8, ls="--")
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].legend(loc="upper left")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Rolling 30-day volatility
    for df in dfs:
        t     = df["Ticker"].iloc[0]
        color = PALETTE.get(t, "#58a6ff")
        vol   = df["Daily_Return"].rolling(30).std() * np.sqrt(252) * 100
        axes[1].plot(df.index, vol, color=color, lw=1.2, label=t)

    axes[1].set_ylabel("Annualised Volatility 30d (%)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "multi_stock_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_correlation_heatmap(dfs: list):
    """Pairwise correlation of daily returns."""
    returns = pd.DataFrame({
        df["Ticker"].iloc[0]: df["Daily_Return"] for df in dfs
    }).dropna()

    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Return Correlation Matrix", fontsize=14,
                 color="#e6edf3", fontweight="bold")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#21262d",
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_summary_dashboard(stats_list: list):
    """Bar chart dashboard of key metrics."""
    df_stats = pd.DataFrame(stats_list)
    tickers  = df_stats["Ticker"].tolist()
    colors   = [PALETTE.get(t, "#58a6ff") for t in tickers]

    metrics = [
        ("Total_Return_%",      "Total Return (%)"),
        ("Annualised_Return_%", "Annualised Return (%)"),
        ("Annual_Volatility_%", "Annual Volatility (%)"),
        ("Sharpe_Ratio",        "Sharpe Ratio"),
        ("Max_Drawdown_%",      "Max Drawdown (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Stock Performance Dashboard", fontsize=18,
                 color="#e6edf3", fontweight="bold")
    axes = axes.flatten()

    for i, (col, title) in enumerate(metrics):
        vals = df_stats[col].tolist()
        bar_colors = ["#f78166" if v < 0 else colors[j]
                      for j, v in enumerate(vals)]
        axes[i].bar(tickers, vals, color=bar_colors, edgecolor="none",
                    width=0.5)
        axes[i].axhline(0, color="#8b949e", lw=0.8)
        axes[i].set_title(title, fontsize=11)
        for j, v in enumerate(vals):
            axes[i].text(j, v + (max(vals) - min(vals)) * 0.02,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                         color="#e6edf3")

    # Hide extra subplot
    axes[-1].axis("off")
    table_data = df_stats[["Ticker", "Start_Price", "End_Price",
                            "Total_Return_%", "Sharpe_Ratio"]].copy()
    table_data.columns = ["Ticker", "Start ($)", "End ($)",
                          "Return (%)", "Sharpe"]
    tbl = axes[-1].table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r > 0 else "#21262d")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#c9d1d9")
    axes[-1].set_title("Summary Table", fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "performance_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def save_data(df: pd.DataFrame):
    ticker = df["Ticker"].iloc[0]
    path   = os.path.join("data", f"{ticker}_analysis.csv")
    df.to_csv(path)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    print("=" * 60)
    print("  Stock Market Analysis System")
    print("  Author: Vishvaa  |  github.com/vishvaa369")
    print("=" * 60)

    all_dfs   = []
    all_stats = []

    for ticker in tickers:
        print(f"\n▶  Processing {ticker} ...")

        # Generate & clean
        df = generate_stock_data(ticker)
        df = clean_data(df)

        # Feature engineering
        df = add_indicators(df)

        # Statistics
        stats = compute_statistics(df)
        all_stats.append(stats)
        print(f"     Total Return : {stats['Total_Return_%']:+.2f}%  |  "
              f"Sharpe: {stats['Sharpe_Ratio']:.2f}  |  "
              f"Max DD: {stats['Max_Drawdown_%']:.2f}%")

        # Per-stock plots
        plot_price_with_ma(df)
        plot_rsi_macd(df)
        plot_returns_distribution(df)

        # Save processed data
        save_data(df)
        all_dfs.append(df)

    # Multi-stock plots
    print("\n▶  Generating comparison charts ...")
    plot_multi_stock_comparison(all_dfs)
    plot_correlation_heatmap(all_dfs)
    plot_summary_dashboard(all_stats)

    # Print summary table
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    df_sum = pd.DataFrame(all_stats).set_index("Ticker")
    print(df_sum[["Total_Return_%", "Annualised_Return_%",
                  "Annual_Volatility_%", "Sharpe_Ratio",
                  "Max_Drawdown_%"]].to_string())
    print("\n✅  Analysis complete. All outputs saved to /outputs/")


if __name__ == "__main__":
    main()
