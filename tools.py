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
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests
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
# Tool 4: latest news headlines
# ---------------------------------------------------------------------------
NEWS_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
NEWS_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def _clean(text: str, limit: int = 300) -> str:
    """Strip HTML tags/extra spaces from a summary and shorten it."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _rss_news(symbol: str, limit: int) -> list[dict]:
    """Yahoo's per-ticker RSS feed: headline, short summary, date, link."""
    r = requests.get(NEWS_RSS_URL, params={"s": symbol, "region": "US", "lang": "en-US"},
                     headers=BROWSER_HEADERS, timeout=15)
    if not r.ok or "<item>" not in r.text:
        return []
    items = []
    for item in ET.fromstring(r.content).iter("item"):
        published = item.findtext("pubDate")
        try:
            published = parsedate_to_datetime(published).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            pass
        items.append({"title": _clean(item.findtext("title"), 200),
                      "summary": _clean(item.findtext("description")),
                      "published": published, "link": item.findtext("link")})
    return items[:limit]


def _search_news(query: str, limit: int) -> list[dict]:
    """Yahoo's search: works for names and topics too (e.g. 'Bitcoin')."""
    r = requests.get(NEWS_SEARCH_URL, params={"q": query, "newsCount": limit, "quotesCount": 0},
                     headers=BROWSER_HEADERS, timeout=15)
    if not r.ok:
        return []
    return [{"title": _clean(n.get("title"), 200), "publisher": n.get("publisher"),
             "published": datetime.fromtimestamp(n["providerPublishTime"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
             if n.get("providerPublishTime") else None, "link": n.get("link")}
            for n in (r.json().get("news") or [])][:limit]


def get_news(symbol: str, limit: int) -> dict:
    """Latest headlines for a symbol (or a company / topic name)."""
    query = symbol.strip()
    try:
        items, source = _rss_news(query.upper(), limit), "Yahoo Finance RSS"
        if not items:
            items, source = _search_news(query, limit), "Yahoo Finance search"
    except (requests.RequestException, ET.ParseError, ValueError) as error:
        raise ToolError(f"News request failed: {error}")
    if not items:
        raise ToolError(f"No news found for '{query}'. Try the company name or a different ticker.")
    return {"query": query, "source": source, "count": len(items), "headlines": items,
            "note": "Headlines and short summaries only - not the full articles."}


# ---------------------------------------------------------------------------
# Tool 5: company fundamentals
# ---------------------------------------------------------------------------
def _big_number(value):
    """5012516765696 -> '5.01T' (easier for people and models to read)."""
    if value is None:
        return None
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"{value / size:.2f}{suffix}"
    return f"{value:,.0f}"


def _pct(value):
    """Yahoo gives margins/growth as fractions: 0.276 -> 27.6."""
    return None if value is None else round(value * 100, 1)


def get_fundamentals(symbol: str) -> dict:
    """Valuation, profitability, growth, balance sheet, analyst view, next earnings."""
    symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception as error:
        raise ToolError(f"Fundamentals request failed for {symbol}: {error}")
    if not info.get("quoteType"):
        raise ToolError(f"No fundamentals for '{symbol}'. Check the Yahoo ticker (e.g. AAPL, MSFT, BTC-USD).")

    result = {
        "symbol": symbol, "name": info.get("longName") or info.get("shortName"),
        "type": info.get("quoteType"), "sector": info.get("sector"), "industry": info.get("industry"),
        "currency": info.get("currency"), "market_cap": _big_number(info.get("marketCap")),
        "valuation": {"pe_trailing": info.get("trailingPE"), "pe_forward": info.get("forwardPE"),
                      "peg_ratio": info.get("pegRatio"), "price_to_book": info.get("priceToBook"),
                      "eps_trailing": info.get("trailingEps"), "eps_forward": info.get("forwardEps")},
        "profitability_pct": {"profit_margin": _pct(info.get("profitMargins")),
                              "operating_margin": _pct(info.get("operatingMargins")),
                              "return_on_equity": _pct(info.get("returnOnEquity"))},
        "growth_pct_yoy": {"revenue": _pct(info.get("revenueGrowth")), "earnings": _pct(info.get("earningsGrowth"))},
        "financials": {"revenue_ttm": _big_number(info.get("totalRevenue")),
                       "free_cash_flow": _big_number(info.get("freeCashflow")),
                       "total_cash": _big_number(info.get("totalCash")),
                       "total_debt": _big_number(info.get("totalDebt")),
                       "debt_to_equity": info.get("debtToEquity")},
        "dividend": {"yield_pct": info.get("dividendYield"),   # already a percentage in current Yahoo data
                     "payout_ratio_pct": _pct(info.get("payoutRatio"))},
        "risk": {"beta": info.get("beta"), "52w_high": info.get("fiftyTwoWeekHigh"),
                 "52w_low": info.get("fiftyTwoWeekLow")},
        "analysts": {"recommendation": info.get("recommendationKey"),
                     "count": info.get("numberOfAnalystOpinions"),
                     "target_mean": info.get("targetMeanPrice"),
                     "target_low": info.get("targetLowPrice"), "target_high": info.get("targetHighPrice")},
        "business_summary": _clean(info.get("longBusinessSummary"), 400) or None,
    }
    try:
        calendar = ticker.calendar or {}
        dates = calendar.get("Earnings Date") or []
        result["next_earnings_date"] = str(dates[0]) if dates else None
        result["ex_dividend_date"] = str(calendar.get("Ex-Dividend Date")) if calendar.get("Ex-Dividend Date") else None
    except Exception:
        result["next_earnings_date"] = None

    # Drop empty fields so crypto / indices (which have few fundamentals) stay short.
    def prune(d):
        return {k: (prune(v) if isinstance(v, dict) else v) for k, v in d.items()
                if v is not None and not (isinstance(v, dict) and not prune(v))}
    result = prune(result)
    result["note"] = "From Yahoo Finance; figures can lag the latest company reports by days."
    return result


# ---------------------------------------------------------------------------
# Tool 6: save the written report
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
        "name": "get_news",
        "description": (
            "Latest news headlines (with short summaries and dates) for a ticker, company name or topic. "
            "Use it to explain why something moved or what's happening around a market. "
            "Headlines only - never claim more than the headline/summary says."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Yahoo ticker (AAPL, BTC-USD) or a name/topic"},
                "limit": {"type": "integer", "enum": [5, 10, 15]},
            },
            "required": ["symbol", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_fundamentals",
        "description": (
            "Company fundamentals for a stock: sector, market cap, P/E (trailing/forward), PEG, price/book, "
            "EPS, margins, revenue/earnings growth, cash, debt, dividend, beta, analyst rating and price "
            "targets, next earnings date, and a short business summary. Crypto and indices return only a "
            "few fields."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Yahoo ticker, e.g. AAPL"}},
            "required": ["symbol"],
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
    "get_news": get_news,
    "get_fundamentals": get_fundamentals,
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
