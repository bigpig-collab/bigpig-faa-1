#!/usr/bin/env python3
"""Forecast Polymarket Yes prices 240 minutes ahead, five quantiles a minute.

Reads every <id>.in case file from $MAKEFUN_INPUT_DIR (else /input/) and
writes one <id>.out forecast to $MAKEFUN_OUTPUT_DIR (else /output/) for
each. Python 3.13 standard library only, no network, deterministic.

Method, per market, from that market's own price history only. This is
v2 of the forecaster; v1 used the house baseline's Gaussian shape with a
per-market sigma and scored 0.276 on the practice. v2 keeps v1's centre
and volatility estimator and changes the shape and the level scaling:

  * Centre: the last price (persistence). Drift is zero. A drift was
    tested thoroughly and does not survive leave-one-case-out
    validation: regressing the future change on the last 5-minute
    change, the last 60-minute change and the distance from the
    24-hour mean gives a multivariate R-squared of 0.26 at 240
    minutes, yet adding that drift at full strength makes the loss
    worse on 5 of the 6 held-out cases, and even at half strength the
    gain is within noise. The median future change is zero for
    essentially every group of markets, so the last price is the
    honest median.
  * Volatility: the average realised per-minute standard deviation
    over the last 30 and the last 120 minutes, clamped to
    [0.0005, 0.004] exactly as in v1. A short window tracks the
    market's current regime (prices move in steps: flat for minutes,
    then a tick or a jump) and averaging two windows steadies it.
    Longer windows, EWMA volatility and jump-boosted volatility were
    all tried and all made the loss worse.
  * Level scaling: the volatility is multiplied by sqrt(price / 0.05)
    and then clamped to [0.0003, 0.004]. A market priced at 0.30
    moves about twice as far in absolute terms as one priced at
    0.05 with the same relative volatility, and the practice bears
    that out: the price level predicts the size of the future move.
  * Shape: an asymmetric, peaked quantile vector
    [-1.4, -0.6, 0, 0.05, 1.644854] in place of the Gaussian
    [-1.644854, -0.67449, 0, 0.67449, 1.644854]. The realised
    distribution of price changes is far more peaked than a
    Gaussian (most minutes the price does not move at all, so the
    25th and 75th percentiles sit almost on the centre) and it is
    skewed down (most candidates lose probability as the field
    narrows, so the upper quantiles are much tighter than the lower
    ones). The vector was chosen by a grid search on the practice
    and confirmed by leave-one-case-out: it beats v1 on 5 of the 6
    held-out cases, and re-tuning it on any 5 cases still picks a
    vector from the same asymmetric family.

  * Quantiles: q_tau(h) = centre + z_tau * sigma * sqrt(h), clamped
    to [0, 1], rounded to 6 decimals, forced non-decreasing.

Practice (NVIDIA CEO event, real answers fetched from Polymarket):
mean pinball loss 0.260881 against v1's 0.276231 and the baseline's
0.363589. Leave-one-case-out mean held-out loss 0.274815 against
v1's 0.285941, so the improvement generalises. The 0.21 target was
not reached with legitimate methods: the centre is the dominant
term and it is news-driven and unpredictable from price history,
and every other legitimate shape, spread, drift, smoothing and
cross-market idea tested made the loss worse or no better.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# v2's quantile vector: asymmetric and peaked (see the module docstring).
# The median stays at the centre; the upper inner quantile sits almost
# on the centre (prices rarely rise far) and the lower side is wider
# (prices fall further than they rise).
Z = [-1.4, -0.6, 0.0, 0.05, 1.644854]

# Bounds on the raw realised volatility, in price units per sqrt minute.
RAW_FLOOR = 0.0005
RAW_CAP = 0.004
# Bounds on the volatility after the price-level scaling.
SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.004
# The price at which the level scaling is 1.
LEVEL_REF = 0.05
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
    short and the long window, clamped, then scaled by the square
    root of the price level relative to LEVEL_REF, then clamped
    again. A high-priced market moves further in absolute terms
    than a low-priced one at the same relative volatility."""
    raw = (sd_changes(prices, WIN_SHORT) + sd_changes(prices, WIN_LONG)) / 2.0
    if raw < RAW_FLOOR:
        raw = RAW_FLOOR
    if raw > RAW_CAP:
        raw = RAW_CAP
    scaled = raw * math.sqrt(prices[-1] / LEVEL_REF)
    if scaled < SIGMA_FLOOR:
        return SIGMA_FLOOR
    if scaled > SIGMA_CAP:
        return SIGMA_CAP
    return scaled


def forecast_market(market, horizon):
    """The 240-row forecast for one market: each row the five quantiles
    for one minute after the checkpoint, around the last price. Rows
    are forced non-decreasing so the output is a valid quantile set
    even where the asymmetric vector and the [0, 1] clamp meet."""
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
        row = [price6(centre + z * sigma * root) for z in Z]
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
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
