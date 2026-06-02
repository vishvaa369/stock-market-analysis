# 📈 Stock Market Analysis System

<p align="center">
  <img src="outputs/performance_dashboard.png" alt="Dashboard" width="800"/>
</p>

> A Python-based stock market analysis system that evaluates historical stock price data, identifies market trends, and generates insightful graphical reports to support investment decision-making.

---

## 🚀 Features

| Category | Details |
|---|---|
| **Data Simulation** | Realistic OHLCV data via Geometric Brownian Motion |
| **Data Cleaning** | Duplicate removal, forward-fill, OHLC validation |
| **Technical Indicators** | MA (20/50/200), Bollinger Bands, RSI, MACD, ATR, OBV |
| **Statistical Analysis** | Sharpe Ratio, Max Drawdown, Annualised Return, Volatility |
| **Visualisations** | 7 chart types per stock + multi-stock comparison |

---

## 🗂 Project Structure

```
stock-market-analysis/
│
├── stock_analysis.py          # Main analysis pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── data/                      # Processed CSV outputs
│   ├── AAPL_analysis.csv
│   ├── GOOGL_analysis.csv
│   ├── MSFT_analysis.csv
│   ├── AMZN_analysis.csv
│   └── TSLA_analysis.csv
│
└── outputs/                   # Generated charts & reports
    ├── performance_dashboard.png
    ├── multi_stock_comparison.png
    ├── correlation_heatmap.png
    ├── AAPL_price_ma.png
    ├── AAPL_rsi_macd.png
    ├── AAPL_returns_distribution.png
    └── ...
```

---

## 📐 Technical Indicators Explained

### Moving Averages
- **MA20** — Short-term trend (swing trading signals)
- **MA50** — Medium-term momentum
- **MA200** — Long-term trend direction; price above = bullish

### Bollinger Bands (20d, 2σ)
Visualise volatility. Price touching the upper band = overbought; lower band = oversold.

### RSI (14-day)
Momentum oscillator. Values > 70 indicate overbought; < 30 indicate oversold conditions.

### MACD (12/26/9)
Trend-following momentum indicator. Bullish signal when MACD crosses above the signal line.

### ATR (14-day)
Average True Range — measures market volatility. Higher ATR = higher risk/reward.

### OBV
On-Balance Volume — confirms trend direction using volume flow.

---

## 📈 Output Metrics

| Metric | Description |
|---|---|
| Total Return % | Overall gain/loss from start to end |
| Annualised Return % | Compound annual growth rate |
| Daily/Annual Volatility | Standard deviation of returns |
| Sharpe Ratio | Risk-adjusted return (higher = better) |
| Max Drawdown % | Largest peak-to-trough decline |
| Best / Worst Day % | Extreme single-day movements |

---

## 👤 Author

**Vishvaa**
- GitHub: [@vishvaa369](https://github.com/vishvaa369)
- Email: vishvaa369@gmail.com

---


---

*Built with Python · pandas · NumPy · Matplotlib · Seaborn*
