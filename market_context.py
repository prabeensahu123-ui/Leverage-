"""
market_context.py
Live context that actually moves BTC: funding, OI, dollar, yields, session.
Used as FILTERS on top of the model, not as extra oscillators.
"""

from datetime import datetime, timezone
import requests

DELTA_BASE = "https://api.india.delta.exchange"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
HEADERS = {"User-Agent": "Mozilla/5.0 LeverageSignal/1.0"}


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_delta_derivatives(symbol="BTCUSD"):
    out = {
        "funding_rate": 0.0,
        "oi_btc": 0.0,
        "oi_usd": 0.0,
        "oi_change_usd_6h": 0.0,
        "mark_price": 0.0,
        "ok": False,
    }
    try:
        r = requests.get(f"{DELTA_BASE}/v2/tickers/{symbol}", timeout=5).json().get("result", {})
        out["funding_rate"] = _f(r.get("funding_rate"))
        out["oi_btc"] = _f(r.get("oi_value", r.get("oi")))
        out["oi_usd"] = _f(r.get("oi_value_usd"))
        out["oi_change_usd_6h"] = _f(r.get("oi_change_usd_6h"))
        out["mark_price"] = _f(r.get("mark_price", r.get("close")))
        out["ok"] = True
    except Exception:
        pass
    return out


def fetch_yahoo_change(symbol):
    """Daily % change from Yahoo chart meta. Returns None on failure."""
    try:
        r = requests.get(
            YAHOO_CHART.format(sym=symbol),
            headers=HEADERS,
            params={"interval": "1d", "range": "5d"},
            timeout=6,
        ).json()
        res = r["chart"]["result"][0]
        meta = res.get("meta", {})
        px = _f(meta.get("regularMarketPrice"))
        prev = _f(meta.get("chartPreviousClose") or meta.get("previousClose"))
        if px and prev:
            return {"price": px, "change_pct": (px - prev) / prev * 100.0}
        closes = (res.get("indicators", {}).get("quote", [{}])[0].get("close") or [])
        closes = [c for c in closes if c is not None]
        if len(closes) >= 2:
            return {"price": closes[-1], "change_pct": (closes[-1] - closes[-2]) / closes[-2] * 100.0}
    except Exception:
        pass
    return None


def session_label(now=None):
    h = (now or datetime.now(timezone.utc)).hour
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 13:
        return "LONDON"
    if 13 <= h < 21:
        return "US"
    return "ASIA"


def build_market_context(symbol="BTCUSD"):
    deriv = fetch_delta_derivatives(symbol)
    dxy = fetch_yahoo_change("DX-Y.NYB")
    tnx = fetch_yahoo_change("%5ETNX")

    funding = deriv["funding_rate"]
    # Delta sometimes stores 0.01 meaning 1.0% ; normal 8h rate is ~0.0001.
    funding_pct = funding * 100.0 if abs(funding) < 0.05 else funding
    oi_chg = deriv["oi_change_usd_6h"]
    dxy_chg = dxy["change_pct"] if dxy else 0.0
    tnx_chg = tnx["change_pct"] if tnx else 0.0
    session = session_label()

    crowded_long = funding_pct >= 0.03 and oi_chg > 0
    crowded_short = funding_pct <= -0.03 and oi_chg > 0
    dollar_strong = dxy_chg >= 0.25 or tnx_chg >= 1.0
    dollar_weak = dxy_chg <= -0.25 or tnx_chg <= -1.0

    veto_buy = []
    veto_sell = []
    if crowded_long:
        veto_buy.append("crowded longs (funding+OI)")
    if crowded_short:
        veto_sell.append("crowded shorts (funding+OI)")
    if dollar_strong:
        veto_buy.append("DXY/yields up")
    if dollar_weak:
        veto_sell.append("DXY/yields down")
    if session == "ASIA":
        # not a hard veto — tag only
        pass

    return {
        "funding_pct": round(funding_pct, 4),
        "oi_btc": deriv["oi_btc"],
        "oi_usd": deriv["oi_usd"],
        "oi_change_usd_6h": oi_chg,
        "dxy": dxy["price"] if dxy else None,
        "dxy_change_pct": round(dxy_chg, 3) if dxy else None,
        "us10y": tnx["price"] if tnx else None,
        "us10y_change_pct": round(tnx_chg, 3) if tnx else None,
        "session": session,
        "crowded_long": crowded_long,
        "crowded_short": crowded_short,
        "dollar_strong": dollar_strong,
        "dollar_weak": dollar_weak,
        "veto_buy": veto_buy,
        "veto_sell": veto_sell,
        "deriv_ok": deriv["ok"],
        "macro_ok": bool(dxy or tnx),
    }


def apply_context_filter(side, ctx):
    """Return (side_or_None, reason)."""
    if side == "BUY" and ctx.get("veto_buy"):
        return None, "; ".join(ctx["veto_buy"])
    if side == "SELL" and ctx.get("veto_sell"):
        return None, "; ".join(ctx["veto_sell"])
    return side, ""
