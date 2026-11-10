#!/usr/bin/env python3
"""Dividend-capture study — market-agnostic pilot script.

Generic design (reusable for any market — EGX, US via Alpaca/Nsave, ...):
  * The ticker symbol, the dividend-event list and the look-ahead window are
    INPUTS (CLI args / JSON file / the COMMANDS workflow variable), never
    hard-coded market assumptions. There is no exchange-calendar logic:
    events are anchored to whatever trading sessions exist in the data.
  * This pilot's defaults are the Cairo Poultry (POUL) values, supplied as
    data below and overridable with --events <json-file>.

Data comes from Yahoo Finance. The download is TIMEZONE-SAFE, generically:
some exchanges have historical local midnights that never existed (days
when DST rules changed — e.g. a spring-forward at midnight), and naive
tz_localize() of such timestamps raises NonExistentTimeError. The fetchers
below therefore (a) ask yfinance for a tz-naive index (ignore_tz=True),
and (b) as a final fallback read Yahoo's chart API directly and derive
session *dates* from the returned epoch instants via tz_convert + .date,
which never localises a (possibly nonexistent) local midnight. No
exchange-specific handling anywhere — this works for any market.

For each dividend event (rights-end date = last cum session, ex-date,
cash dividend per share) the script anchors to the actual sessions in the
downloaded series:
  entry       = close of the last session before the ex-date
  exits       = ex-day open, ex-day close, close N sessions after ex-day
                (N ~= payment day, default 2)
  capture     = dividend - (entry_close - exit_price)
  capture %   = capture / entry_close * 100

Yahoo serves split/merger-adjusted historical prices. When an event list
carries verified exchange entry closes, the script detects the scale
factor (median of downloaded/verified entry closes) and, if it differs
from 1, rescales prices back to the verified scale for the study so the
cash dividends (verified, unadjusted) stay consistent with the prices.
The raw downloaded series is always saved unchanged.

It also runs one vectorbt Portfolio.from_signals across all events
(entries at cum closes, exits at ex-day closes) as the vectorbt showcase.

Outputs (into results/):
  poul_ohlc.csv  raw OHLC series as downloaded
  summary.json   full machine-readable results
  report.md      human-readable report

If no ticker returns data the verdict is written into the results and the
script exits 0 — the loop working plus the data verdict IS the result.
"""

import argparse
import json
import os
import sys

import pandas as pd

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# PILOT INPUT DATA (this run's defaults — data, not code assumptions).
# Verified Cairo Poultry (EGX: POUL) cash-dividend events.
# end = last cum (rights-end) session, ex = ex-date, div = EGP per share,
# verified_entry_close = actual exchange close on the last cum session
# (old, pre-merger share scale) — used only for scale detection.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = ["POUL.CA", "POUL.EY", "POUL"]
DEFAULT_EVENTS = [
    {"label": "2018",     "end": "2018-04-05", "ex": "2018-04-10", "div": 0.25,
     "verified_entry_close": 9.02},
    {"label": "2019",     "end": "2019-07-14", "ex": "2019-07-15", "div": 0.25,
     "verified_entry_close": 4.97},
    {"label": "2021",     "end": "2021-05-17", "ex": "2021-05-18", "div": 0.16,
     "verified_entry_close": 2.93},
    {"label": "2024",     "end": "2024-10-21", "ex": "2024-10-22", "div": 0.847875,
     "verified_entry_close": 12.25},
    {"label": "Apr 2026", "end": "2026-04-26", "ex": "2026-04-27", "div": 0.33,
     "verified_entry_close": 34.91},
]
DEFAULT_PAYDAY_OFFSET = 2   # sessions after ex-day (~ payment day)
HISTORY_START = "2017-01-01"


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Timezone-safe fetching. Strategies are tried in order per ticker; the
# first one returning rows wins. Every attempt is logged and recorded.
# ---------------------------------------------------------------------------

def _clean_frame(df):
    """Normalise a raw provider frame to a tz-naive daily OHLC frame."""
    if df is None or len(df) == 0:
        return None
    # Newer yfinance may return MultiIndex columns for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    if "Close" not in keep:
        return None
    df = df[keep].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        # Convert instants to exchange-local DATES. tz_convert() is safe
        # for every instant; taking .date never localises a midnight, so
        # DST-gap days (nonexistent local midnights) cannot crash this.
        idx = pd.DatetimeIndex(idx.date)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    return df if len(df) else None


def _strat_download_ignore_tz(ticker):
    """Strategy 1: yf.download with ignore_tz=True (skips tz conversion)."""
    import yfinance as yf
    try:
        return yf.download(ticker, start=HISTORY_START, auto_adjust=False,
                           progress=False, ignore_tz=True, threads=False)
    except TypeError:
        # Installed yfinance predates ignore_tz on download().
        return yf.download(ticker, start=HISTORY_START, auto_adjust=False,
                           progress=False, threads=False)


def _strat_ticker_history(ticker):
    """Strategy 2: yf.Ticker().history variants with ignore_tz."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        return t.history(start=HISTORY_START, auto_adjust=False,
                         ignore_tz=True)
    except TypeError:
        return t.history(start=HISTORY_START, auto_adjust=False)


def _strat_chart_api(ticker):
    """Strategy 3: Yahoo chart API directly via requests.

    Builds the OHLC frame from the JSON payload; epoch timestamps are
    converted to session dates in the exchange timezone reported by the
    payload itself (meta.exchangeTimezoneName) via tz_convert + .date —
    yfinance's tz code is bypassed entirely.
    """
    import time

    import requests

    p1 = int(pd.Timestamp(HISTORY_START, tz="UTC").timestamp())
    p2 = int(time.time()) + 86400
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker
    resp = requests.get(
        url,
        params={"period1": p1, "period2": p2, "interval": "1d",
                "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    if not stamps:
        return None
    quote = result["indicators"]["quote"][0]
    meta = result.get("meta", {})
    tz_name = meta.get("exchangeTimezoneName") or "UTC"
    idx_utc = pd.to_datetime(stamps, unit="s", utc=True)
    try:
        dates = idx_utc.tz_convert(tz_name).date
    except Exception:
        dates = idx_utc.date
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=index,
    )


FETCH_STRATEGIES = [
    ("yf_download_ignore_tz", _strat_download_ignore_tz),
    ("yf_ticker_history", _strat_ticker_history),
    ("yahoo_chart_api", _strat_chart_api),
]


def fetch_ohlc(tickers):
    """Try each ticker x strategy in order.

    Returns (ticker, strategy_name, dataframe, attempts).
    """
    attempts = []
    for ticker in tickers:
        for name, fn in FETCH_STRATEGIES:
            try:
                raw = fn(ticker)
            except Exception as exc:  # provider/network error -> next
                attempts.append({"ticker": ticker, "strategy": name,
                                 "rows": 0, "error": str(exc)[:300]})
                log(f"[data] {ticker} via {name}: ERROR {exc}")
                continue
            df = _clean_frame(raw)
            if df is None:
                attempts.append({"ticker": ticker, "strategy": name,
                                 "rows": 0, "error": None})
                log(f"[data] {ticker} via {name}: no rows")
                continue
            attempts.append({"ticker": ticker, "strategy": name,
                             "rows": int(len(df)), "error": None})
            log(f"[data] {ticker} via {name}: {len(df)} rows, "
                f"{df.index[0].date()} -> {df.index[-1].date()}, "
                f"first close {df['Close'].iloc[0]}, "
                f"last close {df['Close'].iloc[-1]}")
            return ticker, name, df, attempts
    return None, None, None, attempts


def detect_price_scale(df, events):
    """Compare downloaded entry closes with verified exchange closes.

    Returns ratios per event, their median, and the multiplier that maps
    downloaded prices back onto the verified scale (1.0 when the series
    already matches the verified closes).
    """
    ratios = {}
    raw_entry_closes = {}
    for e in events:
        verified = e.get("verified_entry_close")
        if not verified:
            continue
        ex_date = pd.Timestamp(e["ex"])
        after = df.index[df.index >= ex_date]
        if len(after) == 0:
            continue
        pos = df.index.get_loc(after[0])
        if pos == 0:
            continue
        entry_close = float(df["Close"].iloc[pos - 1])
        raw_entry_closes[e["label"]] = entry_close
        ratios[e["label"]] = entry_close / float(verified)
    if not ratios:
        return {"ratios": {}, "raw_entry_closes": {},
                "median_ratio": None, "multiplier_applied": 1.0}
    median_ratio = float(pd.Series(list(ratios.values())).median())
    multiplier = 1.0 if 0.97 <= median_ratio <= 1.03 else 1.0 / median_ratio
    return {"ratios": ratios, "raw_entry_closes": raw_entry_closes,
            "median_ratio": median_ratio, "multiplier_applied": multiplier}


def study_event(df, event, payday_offset):
    """Anchor one dividend event to the sessions present in df."""
    ex_date = pd.Timestamp(event["ex"])
    res = {
        "label": event["label"],
        "rights_end_date": event["end"],
        "ex_date": event["ex"],
        "dividend": event["div"],
        "status": "ok",
    }

    after = df.index[df.index >= ex_date]
    if len(after) == 0:
        res["status"] = "no session on/after ex-date in data"
        return res
    ex_ts = after[0]
    ex_pos = df.index.get_loc(ex_ts)
    if ex_pos == 0:
        res["status"] = "no session before ex-date in data"
        return res

    entry_ts = df.index[ex_pos - 1]
    entry_close = float(df["Close"].iloc[ex_pos - 1])
    div = float(event["div"])

    res["entry_session"] = str(entry_ts.date())
    res["entry_close"] = entry_close
    res["ex_session"] = str(ex_ts.date())
    res["dividend_yield_pct"] = div / entry_close * 100.0

    exits = {
        "ex_open": float(df["Open"].iloc[ex_pos]),
        "ex_close": float(df["Close"].iloc[ex_pos]),
    }
    pay_pos = ex_pos + payday_offset
    exits["payday_close"] = (
        float(df["Close"].iloc[pay_pos]) if pay_pos < len(df) else None
    )
    res["payday_session"] = (
        str(df.index[pay_pos].date()) if pay_pos < len(df) else None
    )

    res["exits"] = {}
    for name, price in exits.items():
        if price is None:
            res["exits"][name] = None
        else:
            capture = div - (entry_close - price)
            res["exits"][name] = {
                "price": price,
                "capture": capture,
                "capture_pct": capture / entry_close * 100.0,
            }
    return res


def vectorbt_showcase(df, events):
    """One Portfolio.from_signals across all events (cum close -> ex close)."""
    import vectorbt as vbt

    close = df["Close"]
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    used = 0
    for event in events:
        ex_date = pd.Timestamp(event["ex"])
        after = df.index[df.index >= ex_date]
        if len(after) == 0:
            continue
        ex_pos = df.index.get_loc(after[0])
        if ex_pos == 0:
            continue
        entries.iloc[ex_pos - 1] = True
        exits.iloc[ex_pos] = True
        used += 1

    pf = vbt.Portfolio.from_signals(
        close, entries, exits, init_cash=100_000.0, fees=0.0
    )
    out = {
        "events_used": used,
        "init_cash": 100_000.0,
        "final_value": float(pf.value().iloc[-1]),
        "total_return": float(pf.total_return()),
        "total_return_pct": float(pf.total_return()) * 100.0,
    }
    try:
        out["trade_count"] = int(pf.trades.count())
    except Exception:
        stats = pf.stats()
        out["trade_count"] = int(stats.get("Total Trades", used))
    return out


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def _attempts_line(summary):
    return ", ".join(
        f"{a['ticker']} via {a['strategy']} ({a['rows']} rows)"
        for a in summary.get("tickers_tried", [])
    )


def write_report(path, summary):
    lines = []
    lines.append("# Dividend-capture study — pilot run")
    lines.append("")
    if summary.get("ticker_used") is None:
        lines.append("## Data verdict")
        lines.append("")
        lines.append(summary.get("data_verdict", "No data."))
        lines.append("")
        lines.append("Attempts: " + _attempts_line(summary))
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in summary.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    lines.append(f"- Ticker used: **{summary.get('ticker_used')}**")
    lines.append(f"- Fetch strategy used: **{summary.get('strategy_used')}**")
    lines.append(
        f"- Data: {summary.get('rows')} rows, "
        f"{summary.get('first_date')} → {summary.get('last_date')} "
        f"(first close {summary.get('first_close')}, "
        f"last close {summary.get('last_close')})"
    )
    lines.append("- Attempts: " + _attempts_line(summary))
    lines.append("")
    lines.append("Capture = dividend − (entry close − exit price); "
                 "entry = close of the last session before the ex-date.")
    lines.append("")
    lines.append("| Event | Rights-end | Ex-date | Div | Entry session | Entry close | "
                 "Ex open → capture (% ) | Ex close → capture (%) | Payday close → capture (%) |")
    lines.append("|---|---|---|---:|---|---:|---|---|---|")
    for e in summary["events"]:
        if e["status"] != "ok":
            lines.append(f"| {e['label']} | {e['rights_end_date']} | {e['ex_date']} | "
                         f"{e['dividend']} | — | — | {e['status']} | | |")
            continue
        def cell(name):
            x = e["exits"].get(name)
            if x is None:
                return "—"
            return f"{fmt(x['price'])} → {fmt(x['capture'])} ({fmt(x['capture_pct'], 2)}%)"
        label = e["label"] + (" ⚠" if e.get("scale_consistent") is False else "")
        lines.append(
            f"| {label} | {e['rights_end_date']} | {e['ex_date']} | {e['dividend']} | "
            f"{e['entry_session']} | {fmt(e['entry_close'])} | {cell('ex_open')} | "
            f"{cell('ex_close')} | {cell('payday_close')} |"
        )
    lines.append("")
    lines.append("## Averages (across events with data)")
    lines.append("")
    lines.append("| Exit | Avg capture | Avg capture % | Events |")
    lines.append("|---|---:|---:|---:|")
    for name, a in summary["averages"].items():
        lines.append(f"| {name} | {fmt(a['avg_capture'])} | "
                     f"{fmt(a['avg_capture_pct'], 2)}% | {a['n']} |")
    lines.append("")
    vb = summary.get("vectorbt")
    lines.append("## vectorbt showcase — Portfolio.from_signals "
                 "(entries at cum closes, exits at ex-day closes)")
    lines.append("")
    if vb and "error" not in vb:
        lines.append(f"- Events used: {vb['events_used']}")
        lines.append(f"- Init cash: {vb['init_cash']} → final value: {fmt(vb['final_value'], 2)}")
        lines.append(f"- Total return: {fmt(vb['total_return_pct'], 4)}%")
        lines.append(f"- Trade count: {vb['trade_count']}")
    else:
        lines.append(f"- Not available: {vb.get('error') if vb else 'n/a'}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                    help="Ticker candidates, tried in order (any market).")
    ap.add_argument("--events", default=None,
                    help="Path to a JSON file with the dividend events "
                         "(list of {label, end, ex, div}). Defaults to the "
                         "built-in POUL pilot events.")
    ap.add_argument("--payday-offset", type=int, default=DEFAULT_PAYDAY_OFFSET,
                    help="Sessions after the ex-day used as the ~payment-day exit.")
    args = ap.parse_args()

    events = DEFAULT_EVENTS
    if args.events:
        with open(args.events, encoding="utf-8") as f:
            events = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {"tickers_tried": [], "ticker_used": None, "notes": []}

    ticker, strategy, df, attempts = fetch_ohlc(args.tickers)
    summary["tickers_tried"] = attempts

    if df is None:
        summary["data_verdict"] = (
            "NO DATA: no ticker/strategy combination returned rows from "
            "this runner. Tickers tried: " + ", ".join(args.tickers)
        )
        summary["notes"] = [
            "All fetch strategies (yf.download ignore_tz, Ticker.history, "
            "Yahoo chart API) returned no rows or raised; see attempts in "
            "summary.json and run.log for the per-attempt outcomes.",
        ]
        log("[verdict] " + summary["data_verdict"])
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_report(os.path.join(RESULTS_DIR, "report.md"), summary)
        return 0

    summary["ticker_used"] = ticker
    summary["strategy_used"] = strategy
    summary["rows"] = int(len(df))
    summary["first_date"] = str(df.index[0].date())
    summary["last_date"] = str(df.index[-1].date())
    summary["first_close"] = float(df["Close"].iloc[0])
    summary["last_close"] = float(df["Close"].iloc[-1])
    sample = pd.concat([df.head(3), df.tail(3)])
    summary["sample_closes"] = {
        str(d.date()): float(c) for d, c in zip(sample.index, sample["Close"])
    }
    summary["data_verdict"] = (
        f"OK: {ticker} returned {len(df)} rows via {strategy}."
    )

    # Raw series saved exactly as downloaded (before any rescaling).
    df.to_csv(os.path.join(RESULTS_DIR, "poul_ohlc.csv"))

    # --- Corporate-action scale detection ---------------------------------
    scale = detect_price_scale(df, events)
    summary["price_scale"] = scale
    df_study = df
    if scale["multiplier_applied"] != 1.0:
        df_study = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col in df_study.columns:
                df_study[col] = df_study[col] * scale["multiplier_applied"]
        summary["notes"].append(
            "Yahoo's series is corporate-action adjusted: downloaded entry "
            f"closes are ~{scale['median_ratio']:.4f}x the verified exchange "
            "closes (the Sep 2026 merger multiplied the share count by "
            "~3.7356, i.e. an adjustment factor of ~0.2677). Prices were "
            f"rescaled x{scale['multiplier_applied']:.4f} onto the verified "
            "exchange scale for the study so the verified cash dividends "
            "stay consistent with prices; poul_ohlc.csv keeps the raw "
            "downloaded values."
        )
    elif scale["median_ratio"] is not None:
        summary["notes"].append(
            "Downloaded entry closes match the verified exchange closes "
            f"(median ratio {scale['median_ratio']:.4f}); the series is on "
            "the exchange scale and no rescaling was applied."
        )
    summary["notes"].append(
        "Prices fetched with auto_adjust=False (no dividend adjustment), "
        "which is what a capture study needs."
    )
    summary["notes"].append("No fees, slippage or taxes are modelled.")

    results = [study_event(df_study, e, args.payday_offset) for e in events]
    # Per-event scale consistency: an event whose downloaded/verified
    # entry-close ratio disagrees with the applied scale assumption comes
    # from a provider segment that is adjusted or stale relative to the
    # verified closes — flag it instead of silently trusting its row.
    for res, e in zip(results, events):
        r = scale["ratios"].get(e["label"])
        if r is None:
            continue
        res["scale_ratio_vs_verified"] = r
        res["scale_consistent"] = (
            abs(r * scale["multiplier_applied"] - 1.0) <= 0.05
        )
        if not res["scale_consistent"]:
            res["data_quality_flag"] = (
                "Downloaded prices around this event sit on a different "
                f"scale than the verified exchange closes (ratio {r:.4f}); "
                "that segment of the provider series appears split-adjusted "
                "and/or stale, so this row's captures are not reliable."
            )
            summary["notes"].append(
                f"Event {e['label']}: " + res["data_quality_flag"]
            )
    summary["events"] = results

    averages = {}
    for exit_name in ["ex_open", "ex_close", "payday_close"]:
        caps = [e["exits"][exit_name]["capture"] for e in results
                if e["status"] == "ok" and e["exits"].get(exit_name)]
        pcts = [e["exits"][exit_name]["capture_pct"] for e in results
                if e["status"] == "ok" and e["exits"].get(exit_name)]
        averages[exit_name] = {
            "n": len(caps),
            "avg_capture": (sum(caps) / len(caps)) if caps else None,
            "avg_capture_pct": (sum(pcts) / len(pcts)) if pcts else None,
        }
    summary["averages"] = averages

    try:
        summary["vectorbt"] = vectorbt_showcase(df_study, events)
    except Exception as exc:
        summary["vectorbt"] = {"error": str(exc)}
        log(f"[vectorbt] ERROR {exc}")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(os.path.join(RESULTS_DIR, "report.md"), summary)

    log("[done] results written to results/ (poul_ohlc.csv, summary.json, report.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
