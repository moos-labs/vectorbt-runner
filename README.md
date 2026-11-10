# vectorbt-runner

Runs [vectorbt](https://vectorbt.dev/) analyses on GitHub Actions, where the
commands to execute live in a repository **Actions variable** — not in code.

## The pattern (one repo per tool)

1. **Edit the commands**: repo *Settings → Secrets and variables → Actions →
   Variables* → set the `COMMANDS` variable to whatever shell commands this
   run should execute (multi-line works). The same can be done through the
   GitHub API (`PUT /repos/{owner}/{repo}/actions/variables/COMMANDS`).
2. **Trigger manually**: *Actions* tab → “Run analysis” → *Run workflow*
   (or `POST /repos/{owner}/{repo}/actions/workflows/run.yml/dispatches`).
   The workflow has **no other triggers** — it runs only when dispatched.
3. **Collect results**: the workflow uploads everything under `results/` as
   the `results` build artifact (`if: always()`, so partial results survive
   a failing step). Download it from the run page or via the API.

The workflow itself is generic: checkout → Python 3.11 → `pip install -r
requirements.txt` → run `${{ vars.COMMANDS }}` → upload `results/`.
To run a different analysis, change the variable — not the workflow.

## Multi-market by design

Nothing in the workflow or the runner pattern is tied to one exchange.
Scripts take the **ticker symbol and analysis parameters as inputs** (CLI
args and/or the `COMMANDS` variable), so the same repo can analyse an EGX
stock today and a US stock (e.g. markets available through Nsave or Alpaca)
tomorrow — only the inputs change. There are no exchange-calendar
assumptions in the reusable parts: events anchor to whatever trading
sessions exist in the downloaded data.

## Pilot: Cairo Poultry (POUL) dividend capture

Pilot inputs (defaults in `scripts/poul_dividend_capture.py`, overridable
via `--tickers`, `--events <json>` and `--payday-offset`):

- Tickers tried in order: `POUL.CA`, `POUL.EY`, `POUL` (yfinance,
  `auto_adjust=False`)
- Verified dividend events (rights-end / ex-date / EGP per share):
  2018-04-05 / 2018-04-10 / 0.25 · 2019-07-14 / 2019-07-15 / 0.25 ·
  2021-05-17 / 2021-05-18 / 0.16 · 2024-10-21 / 2024-10-22 / 0.847875 ·
  2026-04-26 / 2026-04-27 / 0.33

Pilot `COMMANDS` value:

```sh
mkdir -p results
python scripts/poul_dividend_capture.py 2>&1 | tee results/run.log
```

The script computes, per event, capture = dividend − (entry close − exit
price) for exits at the ex-day open, ex-day close and ~payment-day close,
plus one `vectorbt.Portfolio.from_signals` run across all events, and
writes `results/poul_ohlc.csv`, `results/summary.json` and
`results/report.md`. If no ticker returns data from the runner, that
verdict is written into the results and the run still succeeds.
