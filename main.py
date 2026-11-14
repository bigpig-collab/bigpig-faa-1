#!/usr/bin/env python3
"""Forecast Polymarket Yes prices 240 minutes ahead, five quantiles a minute.

Reads every <id>.in case file from $MAKEFUN_INPUT_DIR (else /input/) and
writes one <id>.out forecast to $MAKEFUN_OUTPUT_DIR (else /output/) for
each. Python 3.13 standard library only, no network, deterministic.

Method, per market, from that market's own price history only:

  * Centre: the last price (persistence). Recent drift does not predict
    the 240-minute move -- on the practice its sign agrees with the
    realised move less often than a coin flip, and any drift, even
    shrunk, makes the loss worse -- so drift is zero.
  * Spread: a per-market sigma, the average of the per-minute standard
    deviation of price changes over the last 30 and the last 120
    minutes. A short window tracks the market's current regime (prices
    move in steps: flat for minutes, then a tick or a jump), and
    averaging two windows steadies the estimate. It is floored at
    0.0005, so a market that has been flat still carries some spread
    for the jump that may come, and capped at 0.004, so one early
    discovery jump cannot blow the spread up for hours.
  * Shape: the house baseline's Gaussian random walk, q_tau(h) =
    centre + z_tau * sigma * sqrt(h), clamped to [0, 1] and rounded to
    6 decimals. The baseline's fixed sigma of 0.0025 is far too wide
    for most markets (its practice 90% interval covers 95% of outcomes);
    a per-market sigma keeps the same honest shape and lets each market
    set its own width.

Practice (NVIDIA CEO event, real answers fetched from Polymarket):
mean pinball loss 0.276 against the baseline's 0.364.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# Spread bounds, in price units per square-root minute.
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.004
# The two windows (minutes) whose realised volatilities are averaged.
WIN_SHORT = 30
WIN_LONG = 120
# Fallbacks for a market with no usable history.
DEFAULT_PRICE = 0.5
DEFAULT_SIGMA = 0.0025


def clamp01(x):
    """x clamped to [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(clamp01(x) * 1e6) / 1e6 + 0.0


def sd_changes(prices, window):
    """Population sd of the per-minute price changes over the last
    `window` minutes (fewer when the history is shorter). Zero when
    there are fewer than two changes to measure."""
    n = len(prices)
    if n < 2:
        return 0.0
    lo = n - 1 - window
    if lo < 0:
        lo = 0
    diffs = [prices[i + 1] - prices[i] for i in range(lo, n - 1)]
    if len(diffs) < 2:
        return 0.0
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) * (d - mean) for d in diffs) / len(diffs)
    return math.sqrt(var)


def sigma_for(prices):
    """This market's spread: the average realised volatility over the
    short and the long window, floored and capped."""
    s = (sd_changes(prices, WIN_SHORT) + sd_changes(prices, WIN_LONG)) / 2.0
    if s < SIGMA_FLOOR:
        return SIGMA_FLOOR
    if s > SIGMA_CAP:
        return SIGMA_CAP
    return s


def forecast_market(market, horizon):
    """The 240-row forecast for one market: each row the five quantiles
    for one minute after the checkpoint, around the last price."""
    prices = market.get("prices") or []
    if prices:
        centre = prices[-1]
        sigma = sigma_for(prices)
    else:
        centre = DEFAULT_PRICE
        sigma = DEFAULT_SIGMA
    rows = []
    for h in range(1, horizon + 1):
        root = math.sqrt(h)
        rows.append([price6(centre + z * sigma * root) for z in Z])
    return rows


def run_case(path):
    """Read one case file and return its forecast object."""
    with open(path, "r", encoding="utf-8") as fh:
        case = json.load(fh)
    horizon = case["horizon_minutes"]
    want = case["quantiles"]
    if len(want) != len(QUANTILES) or any(a != b for a, b in zip(want, QUANTILES)):
        raise ValueError("case asks for quantiles %r, not %r" % (want, QUANTILES))
    forecasts = []
    for market in case["markets"]:
        forecasts.append({"id": market["id"], "q": forecast_market(market, horizon)})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        result = run_case(os.path.join(INPUT_DIR, name))
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
