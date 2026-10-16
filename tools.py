"""
tools.py - the agent's "hands".

An AI model can't fetch data or save files by itself. It can only ASK for
something to be done. Each tool here is:
  1. a normal Python function that does the real work, and
  2. a description (name + what it does + what inputs it takes) that we send
     to Claude, so it knows the tool exists and how to ask for it.

The agent loop (agent.py) runs whichever function Claude asks for and hands
the result back. Claude never runs code itself - our program stays in control.

SAFETY: these tools can only READ market data and WRITE reports to the
reports/ folder. There is deliberately no tool that can place trades.
"""

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

REPORTS_FOLDER = Path(__file__).parent / "reports"

# yfinance prints its own noisy errors for bad symbols; we report those
# ourselves (as ToolError) so the agent can react, so silence its logging.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Price data fetched during this run, kept in memory so the analysis tools
# can reuse it without downloading again. {"BTC-USD": DataFrame, ...}
_price_cache: dict[str, pd.DataFrame] = {}


class ToolError(Exception):
    """A problem we want to report back to Claude so it can adjust its plan."""


# ---------------------------------------------------------------------------
# Tool 1: fetch price history
# ---------------------------------------------------------------------------
def get_market_data(symbol: str, period: str) -> dict:
    """Download daily prices from Yahoo Finance and return a short overview."""
    symbol = symbol.strip().upper()
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    except Exception as error:  # network problems, Yahoo hiccups, etc.
        raise ToolError(f"Download failed for {symbol}: {error}")
    if df.empty:
        raise ToolError(
            f"No data for '{symbol}'. Check the symbol uses Yahoo Finance format "
            "(e.g. AAPL, BTC-USD, EURUSD=X, GC=F, ^GSPC)."
        )

    _price_cache[symbol] = df
    close = df["Close"]
    # Only a summary goes back to Claude - sending hundreds of raw rows would
    # waste tokens (= money) and add nothing the analysis tools can't provide.
    return {
        "symbol": symbol,
        "period": period,
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "trading_days": len(df),
        "last_close": round(close.iloc[-1], 4),
        "period_change_pct": round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
        "period_high": round(df["High"].max(), 4),
        "period_low": round(df["Low"].min(), 4),
        "last_5_closes": [round(x, 4) for x in close.tail(5)],
        "note": "Data cached - you can now call analyze_symbol / compare_symbols on it.",
    }


# ---------------------------------------------------------------------------
# Tool 2: technical analysis on one symbol
# ---------------------------------------------------------------------------
def _cached(symbol: str) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    if symbol not in _price_cache:
        raise ToolError(f"No data loaded for {symbol}. Call get_market_data first.")
    return _price_cache[symbol]


def _rsi(close: pd.Series, length: int = 14) -> float:
    """Relative Strength Index (Wilder's smoothing). >70 often called overbought, <30 oversold."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] else 100.0


def _atr(df: pd.DataFrame, length: int = 14) -> float:
    """Average True Range: the typical daily price range, including gaps."""
    prev_close = df["Close"].shift()
    true_range = pd.concat([df["High"] - df["Low"],
                            (df["High"] - prev_close).abs(),
                            (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return float(true_range.ewm(alpha=1 / length, adjust=False).mean().iloc[-1])


def _pct_return(close: pd.Series, bars: int):
    """% change over the last `bars` trading days, or None if not enough data."""
    if len(close) <= bars:
        return None
    return round((close.iloc[-1] / close.iloc[-1 - bars] - 1) * 100, 2)


def analyze_symbol(symbol: str) -> dict:
    """Trend, momentum, volatility and drawdown for one symbol."""
    df = _cached(symbol)
    close = df["Close"]
    price = close.iloc[-1]

    sma = {n: close.rolling(n).mean().iloc[-1] for n in (20, 50, 200) if len(close) >= n}
    above = {f"above_sma{n}": bool(price > v) for n, v in sma.items()}

    # A simple, transparent trend label built from the moving averages.
    if 50 in sma and 200 in sma:
        if price > sma[50] > sma[200]:
            trend = "uptrend (price > SMA50 > SMA200)"
        elif price < sma[50] < sma[200]:
            trend = "downtrend (price < SMA50 < SMA200)"
        else:
            trend = "mixed / range"
    else:
        trend = "not enough history for SMA200 - fetch a longer period (1y+)"

    atr = _atr(df)
    daily_returns = close.pct_change().dropna()
    return {
        "symbol": symbol.strip().upper(),
        "last_close": round(price, 4),
        "trend": trend,
        **{f"sma{n}": round(v, 4) for n, v in sma.items()},
        **above,
        "rsi14": round(_rsi(close), 1),
        "atr14": round(atr, 4),
        "atr14_pct_of_price": round(atr / price * 100, 2),
        "annualised_volatility_pct": round(daily_returns.std() * math.sqrt(252) * 100, 1),
        "return_1w_pct": _pct_return(close, 5),
        "return_1m_pct": _pct_return(close, 21),
        "return_3m_pct": _pct_return(close, 63),
        "max_drawdown_pct": round((close / close.cummax() - 1).min() * 100, 2),
        "pct_below_period_high": round((1 - price / df["High"].max()) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Tool 3: compare several symbols
# ---------------------------------------------------------------------------
def compare_symbols(symbols: list[str]) -> dict:
    """Side-by-side returns/volatility plus how closely the symbols move together."""
    if len(symbols) < 2:
        raise ToolError("Give at least two symbols to compare.")
    closes = pd.DataFrame({s.strip().upper(): _cached(s)["Close"] for s in symbols})
    # Different markets trade on different days (crypto trades weekends):
    # line them up by calendar date and keep only days they all traded.
    closes.index = closes.index.tz_localize(None).normalize()
    closes = closes.groupby(level=0).last().dropna()
    if len(closes) < 20:
        raise ToolError("Fewer than 20 shared trading days - fetch a longer period.")

    daily = closes.pct_change().dropna()
    table = {
        s: {
            "return_pct": round((closes[s].iloc[-1] / closes[s].iloc[0] - 1) * 100, 2),
            "annualised_volatility_pct": round(daily[s].std() * math.sqrt(252) * 100, 1),
        }
        for s in closes.columns
    }
    return {
        "shared_days": len(closes),
        "from": str(closes.index[0].date()),
        "to": str(closes.index[-1].date()),
        "performance": table,
        "correlation_of_daily_returns": daily.corr().round(2).to_dict(),
    }


# ---------------------------------------------------------------------------
# Tool 4: save the written report
# ---------------------------------------------------------------------------
def save_report(title: str, markdown: str) -> dict:
    """Save a Markdown report into reports/ and return where it went."""
    REPORTS_FOLDER.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"
    path = REPORTS_FOLDER / f"{datetime.now():%Y-%m-%d_%H%M}_{slug}.md"
    # Add the title as a heading, unless the report already starts with one.
    body = markdown if markdown.lstrip().startswith("#") else f"# {title}\n\n{markdown}"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return {"saved_to": str(path)}


# ---------------------------------------------------------------------------
# The descriptions Claude sees. Good descriptions matter as much as the code:
# they are how the model decides WHICH tool to use and WHAT to pass in.
# "strict": True makes the API guarantee the inputs match these schemas.
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "get_market_data",
        "description": (
            "Download daily price history for one symbol from Yahoo Finance and cache it. "
            "Must be called before analyze_symbol or compare_symbols for that symbol. "
            "Symbols use Yahoo format: stocks AAPL, crypto BTC-USD, forex EURUSD=X, "
            "gold GC=F, oil CL=F, S&P 500 ^GSPC, Nasdaq ^IXIC, DAX ^GDAXI, FTSE ^FTSE. "
            "Use period 1y or longer if you need the 200-day moving average."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Yahoo Finance ticker, e.g. BTC-USD"},
                "period": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"]},
            },
            "required": ["symbol", "period"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "analyze_symbol",
        "description": (
            "Technical snapshot of a symbol already fetched with get_market_data: trend label "
            "from SMA20/50/200, RSI14, ATR14, annualised volatility, 1w/1m/3m returns, "
            "max drawdown and distance from the period high."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "compare_symbols",
        "description": (
            "Compare two or more already-fetched symbols over their shared dates: total return, "
            "annualised volatility, and correlation of daily returns (1 = move together, "
            "0 = unrelated, -1 = opposite)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
            "required": ["symbols"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "save_report",
        "description": "Save the finished summary as a Markdown file in the reports folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short report title"},
                "markdown": {"type": "string", "description": "The full report body in Markdown"},
            },
            "required": ["title", "markdown"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# Maps each tool name to the Python function that runs it.
TOOL_FUNCTIONS = {
    "get_market_data": get_market_data,
    "analyze_symbol": analyze_symbol,
    "compare_symbols": compare_symbols,
    "save_report": save_report,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """
    Run the tool Claude asked for. Returns (result_as_text, is_error).

    Errors are returned to Claude instead of crashing the program - that lets
    the agent notice a mistake (like a wrong symbol) and try something else.
    """
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Unknown tool: {name}", True
    try:
        return json.dumps(function(**tool_input), default=str), False
    except ToolError as error:
        return str(error), True
    except Exception as error:  # unexpected bug - still let the agent carry on
        return f"{type(error).__name__}: {error}", True
