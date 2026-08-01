# grid-backtest

Indicator-driven grid strategy backtester for crypto using RSI + Bollinger Bands.

## Features

- RSI(14) + Bollinger Bands(20, 2.0) strategy
- Auto-generates sample BTC/USDT data if no CSV provided
- Win rate, P/L, max drawdown metrics
- 3-panel chart: Price+BB, RSI, Equity curve
- Pure Python indicators (no TA-Lib C dependency)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Generate sample data + backtest
python grid_backtest.py --symbol BTC/USDT --period 30 --capital 10000

# Use your own data
python grid_backtest.py --data my_data.csv --capital 25000 --fee 0.0005
```

## Strategy

- **BUY**: RSI(14) < 30 AND close < lower Bollinger Band
- **SELL**: RSI(14) > 70 AND close > upper Bollinger Band

## License

MIT
