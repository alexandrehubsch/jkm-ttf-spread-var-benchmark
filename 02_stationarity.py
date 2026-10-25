"""
Phase 2 — Spread mean-reversion validation: ADF / KPSS tests and variance ratios by market regime,
Ornstein-Uhlenbeck calibration.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.adfvalues import mackinnonp
from scipy import stats
from statsmodels.tsa.stattools import kpss

from common import COLORS, FIG, PERIODS, STRESS, load_spread, save

warnings.simplefilter("ignore", InterpolationWarning)

OUT = FIG / "02_stationarity"
DAY = pd.Timedelta(days=1)
SAMPLES = {"Full sample": (None, None),
           "Pre-crisis": ("2019-01-01", pd.Timestamp(STRESS[0]) - DAY), "Stress": STRESS,
           "Post-crisis": (pd.Timestamp(STRESS[1]) + DAY, None)}  # backtest stress window (phase 4)
HORIZONS = (5, 20)
WINDOW = 250
CONCLUSIONS = {(True, False): "stationary", (False, True): "unit root",
               (True, True): "reversion with level shifts", (False, False): "inconclusive"}


def adf(df):
    """ADF with constant on defined changes; missing lags set to 0; lag selection by BIC (Schwert max lag)."""
    max_lag = int(12 * (len(df) / 100) ** 0.25)
    lags = pd.concat({i: df["dX"].shift(i).fillna(0) for i in range(1, max_lag + 1)}, axis=1)
    data = pd.concat([df["dX"], df["spread"].shift(1), lags], axis=1)

    def ols(p, start):
        d = data.iloc[start:, :2 + p].dropna()
        return sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit()

    p = min(range(max_lag + 1), key=lambda p: ols(p, max_lag + 1).bic)  # common sample across all p
    t = ols(p, p + 1).tvalues.iloc[1]  # re-estimation on full sample, matching adfuller
    return t, mackinnonp(t, regression="c"), p  # MacKinnon (1994) asymptotic p-value


def unit_root_tests(df, level=0.05):
    t, p, lags = adf(df)
    kp = kpss(df["spread"], regression="c", nlags="auto", result_object=False)  # Hobijn et al. (1998) lags
    reject = (p < level, kp[1] < level)
    return {"N": len(df), "Lags": lags, "ADF stat": t, "ADF p": p, "ADF reject": reject[0],
            "KPSS stat": kp[0], "KPSS p": kp[1], "KPSS reject": reject[1], "Conclusion": CONCLUSIONS[reject]}


def variance_ratio(dx, q):
    """Lo and MacKinlay (1988) variance ratio on overlapping sums of q changes, heteroskedasticity-robust z*."""
    x = dx.dropna().to_numpy()
    e = x - x.mean()
    s = np.convolve(e, np.ones(q), "valid")  # sums of q consecutive changes
    vr = (s @ s / len(s)) / (q * (e @ e) / len(e))
    delta = [(e[j:] ** 2 * e[:-j] ** 2).sum() / (e @ e) ** 2 for j in range(1, q)]  # delta_j / n
    z = (vr - 1) / np.sqrt(sum((2 * (q - j) / q) ** 2 * d for j, d in zip(range(1, q), delta)))
    return vr, 2 * stats.norm.sf(abs(z))


def pairs(df):
    """Pairs (X_{t-1}, dX_t) where dX_t is defined."""
    return pd.DataFrame({"x": df["spread"].shift(1), "y": df["dX"]}).dropna()


def fit_ou(df):
    """OLS AR(1) on dX_t = a + beta X_{t-1} + e_t, then exact OU parameters (dt = 1 day)."""
    p = pairs(df)
    beta, a = np.polyfit(p["x"], p["y"], 1)  # polyfit returns (slope, constant)
    b = 1 + beta
    theta = -np.log(b) if b < 1 else np.nan
    resid = p["y"] - a - beta * p["x"]
    sigma = resid.std(ddof=2) * np.sqrt(2 * theta / (1 - b**2))
    mu = -a / beta if b < 1 else np.nan
    return {"b": b, "theta": theta, "mu": mu, "sigma": sigma, "Half-life": np.log(2) / theta}


def rolling_mu(df, window=WINDOW):
    """Rolling OU equilibrium level, masked when mean reversion is too slow to identify."""
    p = pairs(df).reindex(df.index)
    roll = dict(window=window, min_periods=int(0.8 * window))
    beta = p["y"].rolling(**roll).cov(p["x"]) / p["x"].rolling(**roll).var()
    mu = p["x"].rolling(**roll).mean() - p["y"].rolling(**roll).mean() / beta
    half_life = np.log(2) / -np.log1p(beta.where(beta < 0))
    return mu.where(half_life < window / 2)


def plot_variance_ratio(df, qs=range(2, 61)):
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for (name, (a, b)), color, ls in zip(list(SAMPLES.items())[1:], [COLORS["blue"], COLORS["red"], COLORS["dark"]], ["-", "--", "-"]):
        ax.plot(list(qs), [variance_ratio(df["dX"].loc[a:b], q)[0] for q in qs], color=color, ls=ls, label=name)
    ax.axhline(1, color="0.5", lw=0.5, ls=":")
    ax.set(xlabel="Horizon $q$ (days)", ylabel=r"$VR(q)$", xlim=(1, qs[-1]), title="Variance ratio of the spread changes by market regime")
    ax.legend(loc="center right")
    save(fig, OUT / "variance_ratio.png")


def plot_rolling_mu(df, mu):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(df["spread"], color=COLORS["blue"], alpha=0.5, lw=0.5, label=r"$X_t$")
    ax.plot(mu, color=COLORS["red"], label=rf"Rolling $\hat\mu$ ({WINDOW}-day window)")
    ax.set(ylabel="USD/MMBtu", ylim=(df["spread"].min() * 1.05, df["spread"].max() * 1.8), xlim=(df.index[0], df.index[-1]),
           title="Spread and rolling Ornstein-Uhlenbeck equilibrium level")
    ax.legend(loc="upper left", ncol=2)
    save(fig, OUT / "rolling_mu.png")


if __name__ == "__main__":
    df = load_spread()
    tests = pd.DataFrame([unit_root_tests(df.loc[a:b]) for a, b in SAMPLES.values()], index=list(SAMPLES))
    vr = pd.DataFrame({k: {f"{c} {q}": v for q in HORIZONS for c, v in zip(["VR", "p"], variance_ratio(df["dX"].loc[a:b], q))}
                       for k, (a, b) in SAMPLES.items()}).T
    ou = pd.DataFrame({k: fit_ou(df.loc[a:b]) for k, (a, b) in (SAMPLES | PERIODS).items()}).T
    pd.set_option("display.width", 200)
    print(tests.round(3).to_string(), "\n")
    print(vr.round(3).to_string(), "\n")
    print(ou.round(3).to_string())
    plot_variance_ratio(df)
    plot_rolling_mu(df, rolling_mu(df))
