#!/usr/bin/env python3
"""grid-backtest — RSI + Bollinger Bands grid strategy backtester for crypto."""
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
BB_PERIOD = 20
BB_STD = 2.0

def generate_sample_data(symbol, days, seed=42):
    n = days * 24
    rng = np.random.default_rng(seed)
    start_price = 60_000.0 if "BTC" in symbol.upper() else 100.0
    mu, sigma = 0.00002, 0.012
    shocks = rng.normal(mu, sigma, n)
    close = start_price * np.exp(np.cumsum(shocks))
    open_ = np.empty(n); open_[0] = start_price; open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, sigma * 0.6, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.uniform(20, 400, n) * (1 + np.abs(shocks) * 40)
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="h")
    return pd.DataFrame({"timestamp": idx, "open": open_.round(2), "high": high.round(2),
        "low": low.round(2), "close": close.round(2), "volume": volume.round(4)})

def load_data(path, symbol, days):
    if path and path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"[data] loaded {len(df):,} rows from {path}")
    else:
        df = generate_sample_data(symbol, days)
        out = path or Path("data") / f"{symbol.replace('/', '')}_{days}d.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[data] generated {len(df):,} sample rows → {out}")
    return df.sort_values("timestamp").reset_index(drop=True)

def add_indicators(df):
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    return df

@dataclass
class Trade:
    entry_time: pd.Timestamp; entry_price: float; qty: float; cost: float
    exit_time: Optional[pd.Timestamp] = None; exit_price: Optional[float] = None
    pnl: float = 0.0; pnl_pct: float = 0.0; status: str = "OPEN"

@dataclass
class BacktestResult:
    df: pd.DataFrame; trades: list; equity: pd.Series; initial_capital: float; fee: float

def run_backtest(df, capital, fee):
    cash, qty, cost_basis = capital, 0.0, 0.0
    entry_time, entry_price = None, 0.0
    trades = []
    equity = np.empty(len(df))
    buy_sig = (df["rsi"] < RSI_OVERSOLD) & (df["close"] < df["bb_lower"])
    sell_sig = (df["rsi"] > RSI_OVERBOUGHT) & (df["close"] > df["bb_upper"])
    for i, row in enumerate(df.itertuples()):
        price = float(row.close)
        if buy_sig.iat[i] and qty == 0.0 and pd.notna(row.bb_lower):
            qty = cash / (price * (1 + fee)); cost_basis = cash; cash = 0.0
            entry_time, entry_price = row.timestamp, price
        elif sell_sig.iat[i] and qty > 0.0 and pd.notna(row.bb_upper):
            proceeds = qty * price * (1 - fee); pnl = proceeds - cost_basis
            trades.append(Trade(entry_time, entry_price, qty, cost_basis,
                row.timestamp, price, pnl, pnl / cost_basis * 100.0, "CLOSED"))
            cash, qty = proceeds, 0.0
        equity[i] = cash + qty * price
    if qty > 0.0:
        last = df.iloc[-1]; proceeds = qty * float(last["close"]) * (1 - fee); pnl = proceeds - cost_basis
        trades.append(Trade(entry_time, entry_price, qty, cost_basis,
            last["timestamp"], float(last["close"]), pnl, pnl / cost_basis * 100.0, "OPEN (EOD)"))
    return BacktestResult(df, trades, pd.Series(equity, index=df["timestamp"], name="equity"), capital, fee)

@dataclass
class Metrics:
    total_trades: int; wins: int; losses: int; win_rate: float
    total_pnl: float; total_return_pct: float; final_equity: float
    max_drawdown_pct: float; best_trade: float; worst_trade: float

def compute_metrics(res):
    pnls = [t.pnl for t in res.trades]; wins = sum(1 for p in pnls if p > 0)
    final = float(res.equity.iloc[-1])
    drawdown = (res.equity / res.equity.cummax() - 1.0) * 100.0
    return Metrics(total_trades=len(pnls), wins=wins, losses=len(pnls)-wins,
        win_rate=(wins/len(pnls)*100.0) if pnls else 0.0, total_pnl=final-res.initial_capital,
        total_return_pct=(final/res.initial_capital-1.0)*100.0, final_equity=final,
        max_drawdown_pct=float(drawdown.min()), best_trade=max(pnls, default=0.0),
        worst_trade=min(pnls, default=0.0))

def print_report(symbol, res, m):
    w = 64
    print("\n" + "═" * w)
    print(f" GRID-BACKTEST REPORT — {symbol} ".center(w))
    print("═" * w)
    for label, value in [("Initial capital", f"${res.initial_capital:,.2f}"),
        ("Final equity", f"${m.final_equity:,.2f}"),
        ("Total P/L", f"${m.total_pnl:+,.2f}  ({m.total_return_pct:+.2f}%)"),
        ("Max drawdown", f"{m.max_drawdown_pct:.2f}%"),
        ("Total trades", str(m.total_trades)),
        ("Wins / Losses", f"{m.wins} / {m.losses}"),
        ("Win rate", f"{m.win_rate:.2f}%"),
        ("Best / Worst", f"${m.best_trade:+,.2f} / ${m.worst_trade:+,.2f}"),
        ("Fee rate", f"{res.fee:.3%}")]:
        print(f"  {label:<22}{value:>36}")
    print("─" * w)
    if not res.trades:
        print("  No trades executed."); print("═" * w); return
    print(f"  {'#':<4}{'Entry time':<19}{'Entry$':>11}{'Exit$':>11}{'P/L$':>12}{'P/L%':>8}")
    print("  " + "─" * 60)
    for i, t in enumerate(res.trades[:15], 1):
        et = t.entry_time.strftime("%Y-%m-%d %H:%M")
        print(f"  {i:<4}{et:<19}{t.entry_price:>11,.2f}{(t.exit_price or 0):>11,.2f}{t.pnl:>+12,.2f}{t.pnl_pct:>+7.2f}%")
    if len(res.trades) > 15: print(f"  … and {len(res.trades)-15} more")
    print("═" * w)

def plot_report(res, symbol, out_dir):
    df = res.df; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{symbol.replace('/', '')}_grid_backtest.png"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1.4]})
    ts = df["timestamp"]
    ax1.plot(ts, df["close"], lw=1.0, color="#222", label="Close")
    ax1.plot(ts, df["bb_upper"], lw=0.8, color="#888", ls="--")
    ax1.plot(ts, df["bb_mid"], lw=0.8, color="#aaa", ls=":")
    ax1.plot(ts, df["bb_lower"], lw=0.8, color="#888", ls="--")
    ax1.fill_between(ts, df["bb_lower"], df["bb_upper"], color="#9ecae1", alpha=0.25)
    entries = [(t.entry_time, t.entry_price) for t in res.trades]
    exits = [(t.exit_time, t.exit_price) for t in res.trades if t.exit_time]
    if entries: ax1.scatter(*zip(*entries), marker="^", color="#1a9850", s=95, zorder=5, label="Buy")
    if exits: ax1.scatter(*zip(*exits), marker="v", color="#d73027", s=95, zorder=5, label="Sell")
    ax1.set_ylabel("Price (USDT)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)
    ax2.plot(ts, df["rsi"], lw=1.0, color="#756bb1")
    ax2.axhline(RSI_OVERBOUGHT, color="#d73027", lw=0.8, ls="--")
    ax2.axhline(RSI_OVERSOLD, color="#1a9850", lw=0.8, ls="--")
    ax2.fill_between(ts, RSI_OVERSOLD, RSI_OVERBOUGHT, color="#eee")
    ax2.set_ylim(0, 100); ax2.set_ylabel(f"RSI({RSI_PERIOD})"); ax2.grid(alpha=0.25)
    ax3.plot(res.equity.index, res.equity, lw=1.1, color="#08519c")
    ax3.axhline(res.initial_capital, color="#666", lw=0.8, ls="--")
    ax3.set_ylabel("Equity (USDT)"); ax3.grid(alpha=0.25)
    fig.suptitle(f"grid-backtest — {symbol} | RSI({RSI_PERIOD}) + BB({BB_PERIOD}, {BB_STD})", fontsize=13, fontweight="bold")
    fig.autofmt_xdate(); fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150); plt.close(fig)
    return out

def main(argv=None):
    p = argparse.ArgumentParser(prog="grid-backtest", description="RSI + BB grid strategy backtester.")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--period", type=int, default=30)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--output", type=Path, default=Path("reports"))
    args = p.parse_args(argv)
    print("═" * 64); print(" grid-backtest ".center(64)); print("═" * 64)
    print(f"  symbol={args.symbol}  period={args.period}d  capital=${args.capital:,.2f}  fee={args.fee:.3%}")
    df = load_data(args.data, args.symbol, args.period)
    if len(df) < BB_PERIOD + 5:
        print(f"[error] need at least {BB_PERIOD+5} rows, got {len(df)}", file=sys.stderr); return 1
    df = add_indicators(df)
    n_buy = int(((df["rsi"] < RSI_OVERSOLD) & (df["close"] < df["bb_lower"])).sum())
    n_sell = int(((df["rsi"] > RSI_OVERBOUGHT) & (df["close"] > df["bb_upper"])).sum())
    print(f"[signals] entry={n_buy}  exit={n_sell}")
    res = run_backtest(df, args.capital, args.fee)
    metrics = compute_metrics(res)
    print_report(args.symbol, res, metrics)
    chart = plot_report(res, args.symbol, args.output)
    print(f"[chart] saved → {chart}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
