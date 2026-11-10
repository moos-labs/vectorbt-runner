#!/usr/bin/env python3
"""Dividend-capture study — market-agnostic pilot script.

Generic design (reusable for any market — EGX, US via Alpaca/Nsave, ...):
  * The ticker symbol, the dividend-event list and the look-ahead window are
    INPUTS (CLI args / JSON file / the COMMANDS workflow variable), never
    hard-coded market assumptions. There is no exchange-calendar logic:
    events are anchored to whatever trading sessions exist in the data.
  * This pilot's defaults are the Cairo Poultry (POUL) values, supplied as
    data below and overridable with --events <json-file>.

Data comes from yfinance (this pilot's provider; the study itself only
needs an OHLC frame, so another provider can be swapped in later).

For each dividend event (rights-end date = last cum session, ex-date,
cash dividend per share) the script anchors to the actual sessions in the
downloaded series:
  entry       = close of the last session before the ex-date
  exits       = ex-day open, ex-day close, close N sessions after ex-day
                (N ~= payment day, default 2)
  capture     = dividend - (entry_close - exit_price)
  capture %   = capture / entry_close * 100

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
# end = last cum (rights-end) session, ex = ex-date, div = EGP per share.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = ["POUL.CA", "POUL.EY", "POUL"]
DEFAULT_EVENTS = [
    {"label": "2018",     "end": "2018-04-05", "ex": "2018-04-10", "div": 0.25},
    {"label": "2019",     "end": "2019-07-14", "ex": "2019-07-15", "div": 0.25},
    {"label": "2021",     "end": "2021-05-17", "ex": "2021-05-18", "div": 0.16},
    {"label": "2024",     "end": "2024-10-21", "ex": "2024-10-22", "div": 0.847875},
    {"label": "Apr 2026", "end": "2026-04-26", "ex": "2026-04-27", "div": 0.33},
]
DEFAULT_PAYDAY_OFFSET = 2   # sessions after ex-day (~ payment day)
HISTORY_START = "2017-01-01"


def log(msg):
    print(msg, flush=True)


def fetch_ohlc(tickers):
    """Try each ticker via yfinance; return (ticker, dataframe, attempts)."""
    import yfinance as yf

    attempts = []
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=HISTORY_START,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # provider/network error -> try next ticker
            attempts.append({"ticker": ticker, "rows": 0, "error": str(exc)})
            log(f"[data] {ticker}: ERROR {exc}")
            continue

        if df is None or len(df) == 0:
            attempts.append({"ticker": ticker, "rows": 0, "error": None})
            log(f"[data] {ticker}: no rows")
            continue

        # Newer yfinance may return MultiIndex columns for a single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"]
                 if c in df.columns]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["Close"])

        attempts.append({"ticker": ticker, "rows": int(len(df)), "error": None})
        log(f"[data] {ticker}: {len(df)} rows, "
            f"{df.index[0].date()} -> {df.index[-1].date()}, "
            f"first close {df['Close'].iloc[0]}, last close {df['Close'].iloc[-1]}")
        return ticker, df, attempts

    return None, None, attempts


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


def write_report(path, summary):
    lines = []
    lines.append("# Dividend-capture study — pilot run")
    lines.append("")
    lines.append(f"- Ticker used: **{summary.get('ticker_used')}**")
    lines.append(
        f"- Data: {summary.get('rows')} rows, "
        f"{summary.get('first_date')} → {summary.get('last_date')} "
        f"(first close {summary.get('first_close')}, "
        f"last close {summary.get('last_close')})"
    )
    lines.append(f"- Tickers tried: "
                 + ", ".join(f"{a['ticker']} ({a['rows']} rows)"
                             for a in summary["tickers_tried"]))
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
        lines.append(
            f"| {e['label']} | {e['rights_end_date']} | {e['ex_date']} | {e['dividend']} | "
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
    summary = {
        "tickers_tried": [],
        "ticker_used": None,
        "notes": [
            "Yahoo's series may be adjusted for corporate actions — for POUL "
            "that includes the Sep 2026 merger share-count change — so "
            "pre-merger absolute prices may differ from exchange records, "
            "while percentage captures remain comparable.",
            "Prices are as downloaded with auto_adjust=False (no dividend "
            "adjustment), which is what a capture study needs.",
            "No fees, slippage or taxes are modelled.",
        ],
    }

    ticker, df, attempts = fetch_ohlc(args.tickers)
    summary["tickers_tried"] = attempts

    if df is None:
        summary["data_verdict"] = (
            "NO DATA: none of the tickers returned rows from this runner. "
            "Tickers tried: " + ", ".join(args.tickers)
        )
        log("[verdict] " + summary["data_verdict"])
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_report(os.path.join(RESULTS_DIR, "report.md"), summary)
        return 0

    summary["ticker_used"] = ticker
    summary["rows"] = int(len(df))
    summary["first_date"] = str(df.index[0].date())
    summary["last_date"] = str(df.index[-1].date())
    summary["first_close"] = float(df["Close"].iloc[0])
    summary["last_close"] = float(df["Close"].iloc[-1])
    summary["data_verdict"] = f"OK: {ticker} returned {len(df)} rows."

    df.to_csv(os.path.join(RESULTS_DIR, "poul_ohlc.csv"))

    results = [study_event(df, e, args.payday_offset) for e in events]
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
        summary["vectorbt"] = vectorbt_showcase(df, events)
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
