"""
Phase 3 — One-day-ahead VaR forecasts: EWMA, EWMA-FHS, and three Markov-Switching Ornstein-Uhlenbeck (MS-OU) specifications.
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import chi2
from scipy.special import gammaln, ndtr, ndtri, stdtr  # ln Gamma, Phi, Phi^{-1}, Student-t CDF

from common import COLORS, FIG, FORECASTS, RESULTS, load_spread, save

OUT = FIG / "03_var_models"
START = "2019-01-01"
LEVELS = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
LAMBDA = 0.94
WINDOW = 250
WORKERS = max(1, os.cpu_count() // 4)
SUBPERIODS = {"2019-20": ("2019", "2020"), "2021-22": ("2021", "2022"), "2023-26": ("2023", "2026")}
FIXED = {"both": [], "calm": [1, 3], "stress": [0, 2], "none": [0, 1, 2, 3]}  # (theta, mu) fixed to 0: drift in both regimes, regime 1, regime 2, none


def ewma_variance(dx, lam=LAMBDA):
    """sigma_t^2 given F_{t-1}, initialized to 2018 sample variance; no update following missing change."""
    x = dx.to_numpy()
    s2 = np.empty(len(x))
    s2[0] = dx.loc["2018"].var()
    for t in range(1, len(x)):
        s2[t] = s2[t - 1] if np.isnan(x[t - 1]) else lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(s2, index=dx.index)


def ewma_quantiles(s2):
    return pd.DataFrame({k: ndtri(p) * np.sqrt(s2) for k, p in LEVELS.items()})


def fhs_quantiles(dx, s2, window=WINDOW):
    """Empirical quantiles of standardized residuals from preceding `window` days, rescaled by sigma_t."""
    roll = (dx / np.sqrt(s2)).rolling(window, min_periods=int(0.8 * window))
    return pd.DataFrame({k: roll.quantile(p).shift(1) * np.sqrt(s2) for k, p in LEVELS.items()})  # window closed at t-1


def fhs_pit(dx, s2, window=WINDOW):
    """Rank of epsilon_t among residuals from preceding `window` days: (#{eps <= eps_t} + 1/2) / (n + 1)."""
    eps = (dx / np.sqrt(s2)).to_numpy()
    pit = np.full(len(eps), np.nan)
    for t in range(window, len(eps)):
        w = eps[t - window:t]
        w = w[~np.isnan(w)]
        pit[t] = (np.sum(w <= eps[t]) + 0.5) / (len(w) + 1)
    return pd.Series(pit, index=dx.index)


def std_pdf(z, nu):
    """Standard normal PDF if nu is None, otherwise Student-t PDF with nu dof (unit scale, variance nu / (nu - 2))."""
    if nu is None:
        return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(z**2 / nu))


def std_cdf(z, nu):
    return ndtr(z) if nu is None else stdtr(nu, z)


def ms_moments(params, x_lag, a=None):
    """
    Drift m_{s,t}, scale c_{s,t}, and degrees of freedom of dX_t in each regime.
    params = (theta_1, theta_2, mu_1, mu_2, scale_1, scale_2, p_11, p_22[, nu]).
    a = None: constant OU scale, c_s = sigma_s sqrt((1 - e^{-2 theta_s}) / (2 theta_s)); otherwise c_{s,t} = k_s a_t.
    Student-t innovations if nu is present, normal otherwise.
    """
    theta, mu, scale = params[0:2], params[2:4], params[4:6]
    m = (1 - np.exp(-theta)) * (mu - x_lag[:, None])
    nu = params[8] if len(params) > 8 else None
    if a is None:
        shrink = np.where(theta > 0, -np.expm1(-2 * theta) / (2 * np.maximum(theta, 1e-12)), 1.0)  # limit 1 as theta -> 0
        return m, np.broadcast_to(scale * np.sqrt(shrink), m.shape), nu
    return m, scale * a[:, None], nu


def hamilton(params, x_lag, dx, a=None):
    """Hamilton filter: P(S_t = 2 | F_{t-1}) and log-likelihood; prediction step only when dX_t is missing."""
    m, c, nu = ms_moments(params, x_lag, a)
    f = std_pdf((dx[:, None] - m) / c, nu) / c
    p11, p22 = params[6:8]
    pi2 = (1 - p11) / (2 - p11 - p22)  # stationary distribution of Markov chain
    pred, loglik = np.empty(len(dx)), 0.0
    for t, (f1, f2) in enumerate(f.tolist()):
        q = (1 - pi2) * (1 - p11) + pi2 * p22
        pred[t] = q
        if math.isnan(f1):
            pi2 = q
            continue
        lik = max((1 - q) * f1 + q * f2, 1e-300)  # safeguard against log(0)
        loglik += math.log(lik)
        pi2 = q * f2 / lik
    return pred, loglik


def start_values(x_lag, dx, a=None, student=False):
    ok = ~np.isnan(x_lag) & ~np.isnan(dx)
    beta, _ = np.polyfit(x_lag[ok], dx[ok], 1)
    theta = np.clip(-np.log1p(min(beta, -1e-3)), 0.005, 0.5)
    mu, sx = x_lag[ok].mean(), x_lag[ok].std()
    sd = (dx[ok] if a is None else dx[ok] / a[ok]).std()
    return [np.r_[theta, theta, mu, mu + shift * sx, k1 * sd, ratio, 0.97, 0.97, [6.0] if student else []]
            for k1, ratio, shift in [(0.5, 4.0, 0.0), (0.3, 10.0, -1.0)]]


def fit_ms_ou(x_lag, dx, a=None, student=False, drift="both"):
    """
    Maximum likelihood (L-BFGS-B, two starting points). mu is bounded by observed range of X, p_ss >= 0.8;
    regime 2 scale is regime 1 scale multiplied by r >= 1, ensuring regime 1 is the lower-volatility state.
    drift: regimes with OU drift ("both", "calm" = regime 1, "stress" = regime 2, "none").
    """
    x = x_lag[~np.isnan(x_lag)]
    bounds = [(1e-3, 1.0)] * 2 + [(x.min(), x.max())] * 2 + [(1e-3, 20.0), (1.0, 100.0)] + [(0.8, 0.9995)] * 2
    bounds += [(2.5, 50.0)] if student else []
    for i in FIXED[drift]:
        bounds[i] = (0.0, 0.0)
    lo, hi = np.array(bounds).T
    scales = lambda q: np.r_[q[:5], q[4] * q[5], q[6:]]  # (scale_1, r) -> (scale_1, scale_2)
    fits = [minimize(lambda q: -hamilton(scales(q), x_lag, dx, a)[1], np.clip(q0, lo, hi), method="L-BFGS-B", bounds=bounds)
            for q0 in start_values(x_lag, dx, a, student)]
    return scales(min(fits, key=lambda r: r.fun).x)


def select_drift(x_lag, dx, a=None, student=False, level=0.05):
    """
    Selected drifts by sequential likelihood ratio tests (chi2(2)): absence of mean reversion in high-scale
    regime (phase 2), then remaining drift; a drift is retained only if significant at threshold `level`.
    """
    ll = {d: hamilton(fit_ms_ou(x_lag, dx, a, student, d), x_lag, dx, a)[1] for d in FIXED}
    p = lambda h1, h0: chi2.sf(2 * (ll[h1] - ll[h0]), 2)
    tests = {"Drift regime 2": p("both", "calm")}
    if tests["Drift regime 2"] >= level:
        tests["Drift regime 1"] = p("calm", "none")
        return ("calm" if tests["Drift regime 1"] < level else "none"), tests
    tests["Drift regime 1"] = p("both", "stress")
    return ("both" if tests["Drift regime 1"] < level else "stress"), tests


def mixture_quantile(p, w, m, c, nu):
    lo, hi = (m - 20 * c).min(), (m + 20 * c).max()
    return brentq(lambda x: w @ std_cdf((x - m) / c, nu) - p, lo, hi)  # root of F_t(x) = p over bracketing interval


def ms_ou_forecast(df, a=None, student=False, drift="both"):
    """
    Walk-forward estimation: monthly re-estimation on expanding window, daily filtering with fixed parameters within month.
    Returns forecasted quantiles, predicted probability of regime 2, and PIT F_t(dX_t).
    """
    x_lag, dx = df["spread"].shift(1).to_numpy(), df["dX"].to_numpy()
    cuts = df.index.searchsorted(pd.date_range(START, df.index[-1], freq="MS"))
    n = len(cuts)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        params = list(pool.map(fit_ms_ou, [x_lag[:c] for c in cuts], [dx[:c] for c in cuts],
                               [None if a is None else a[:c] for c in cuts], [student] * n, [drift] * n))
    rows, extra = [], []
    for c0, c1, p in zip(cuts, list(cuts[1:]) + [len(df)], params):
        a_c = None if a is None else a[:c1]
        pred, _ = hamilton(p, x_lag[:c1], dx[:c1], a_c)
        m, c, nu = ms_moments(p, x_lag[:c1], a_c)
        for t in range(c0, c1):
            w = np.array([1 - pred[t], pred[t]])
            rows.append([mixture_quantile(q, w, m[t], c[t], nu) for q in LEVELS.values()])
            extra.append([pred[t], w @ std_cdf((dx[t] - m[t]) / c[t], nu)])
    index = df.index[cuts[0]:]
    extra = pd.DataFrame(extra, index=index, columns=["p_stress", "pit"])
    return pd.DataFrame(rows, index=index, columns=list(LEVELS)), extra


def in_sample_pit(p, x_lag, dx, a=None):
    pred, _ = hamilton(p, x_lag, dx, a)
    m, c, nu = ms_moments(p, x_lag, a)
    return (np.c_[1 - pred, pred] * std_cdf((dx[:, None] - m) / c, nu)).sum(1)


def ms_ou_table(p):
    theta, scale, stay = p[0:2], p[4:6], p[6:8]
    drift = theta > 0  # driftless regime: random walk, mu and half-life undefined
    table = pd.DataFrame({"theta": theta, "mu": np.where(drift, p[2:4], np.nan), "scale": scale,
                          "Half-life": np.log(2) / np.where(drift, theta, np.nan),
                          "p_ss": stay, "Duration": 1 / (1 - stay)}, index=["Regime 1", "Regime 2"])
    return table.assign(nu=p[8]) if len(p) > 8 else table


def information_criteria(x_lag, dx, s2, fits):
    ok = ~np.isnan(x_lag) & ~np.isnan(dx)
    beta, a = np.polyfit(x_lag[ok], dx[ok], 1)
    resid = dx[ok] - a - beta * x_lag[ok]
    n = ok.sum()
    ll = {"OU": -0.5 * n * (np.log(2 * np.pi * resid.var()) + 1),
          "EWMA": -0.5 * np.sum(np.log(2 * np.pi * s2[ok]) + dx[ok] ** 2 / s2[ok])}
    k = {"OU": 3, "EWMA": 0}
    for name, (p, scale, drift) in fits.items():
        ll[name], k[name] = hamilton(p, x_lag, dx, scale)[1], len(p) - len(FIXED[drift])
    return pd.DataFrame({"Log-lik": ll, "Params": k, "BIC": {m: k[m] * np.log(n) - 2 * ll[m] for m in ll}})


def calibration(pits):
    """Moments of z_t = Phi^{-1}(PIT) (N(0,1) under proper calibration) and share of PITs outside [1%, 99%]."""
    rows = {}
    for name, u in pits.items():
        z = pd.Series(ndtri(u.clip(1e-6, 1 - 1e-6)), index=u.index)
        rows[name] = {"Mean z": z.mean(), "Sd z": z.std()} | {f"Sd z {k}": z.loc[a:b].std() for k, (a, b) in SUBPERIODS.items()} \
            | {"Tails": ((u < 0.01) | (u > 0.99)).mean()}
    return pd.DataFrame(rows).T


def plot_regimes(df, p_const, p_level):
    x = df["spread"].loc[START:]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.4), sharex=True)
    axes[0].plot(x, color=COLORS["dark"])
    axes[0].fill_between(x.index, x.min(), x.max(), where=(p_level > 0.5).reindex(x.index, fill_value=False),
                         color=COLORS["light_red"], lw=0, label=r"$P(S_t = 2 \mid \mathcal{F}_{t-1}) > 0.5$, level vol.")
    axes[0].set(ylabel=r"$X_t$ (USD/MMBtu)", title="(a) Spread and stress regime of the level-volatility MS-OU")
    axes[0].legend(loc="lower left", frameon=True, facecolor="white", edgecolor="0.7", framealpha=1)
    for ax, p, title in [(axes[1], p_const, "(b) Constant-volatility MS-OU"), (axes[2], p_level, "(c) Level-volatility MS-OU")]:
        ax.plot(p, color=COLORS["blue"], lw=0.6)
        ax.set(ylabel=r"$P(S_t = 2 \mid \mathcal{F}_{t-1})$", ylim=(0, 1), title=title)
    axes[2].set_xlim(x.index[0], x.index[-1])
    save(fig, OUT / "ms_ou_regimes.png")


def plot_pit(pits, table):
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.2), sharey=True)
    for ax, (name, u) in zip(axes.flat, pits.items()):
        ax.hist(u, bins=20, range=(0, 1), density=True, color=COLORS["light"])
        ax.axhline(1, color=COLORS["red"], ls="--", lw=0.7)
        ax.set(title=name, ylim=(0, 2.6))
        ax.text(0.97, 0.97, f"sd($z$) = {table.loc[name, 'Sd z']:.2f}\n" f"tails = {100 * table.loc[name, 'Tails']:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, bbox={"facecolor": "white", "edgecolor": "0.7", "lw": 0.5})
    fig.delaxes(axes[1, 2])
    for ax in [axes[1, 0], axes[1, 1], axes[0, 2]]:
        ax.set_xlabel(r"PIT $u_t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.tight_layout()
    save(fig, OUT / "pit.png")


if __name__ == "__main__":
    df = load_spread()
    dx_s = df["dX"]
    s2 = ewma_variance(dx_s)
    level = ((df["ttf"] + df["jkm"]) / 2).shift(1).to_numpy()
    evaluable = df.index[(df.index >= START) & dx_s.notna()]
    x_lag, dx = df["spread"].shift(1).to_numpy(), dx_s.to_numpy()
    ms_specs = {"MS-OU constant vol.": (None, False), "MS-OU level vol.": (level, True), "MS-OU EWMA vol.": (np.sqrt(s2.to_numpy()), False)}
    drifts, drift_tests = {}, {}
    for name, (a, student) in ms_specs.items():
        drifts[name], drift_tests[name] = select_drift(x_lag, dx, a, student)  # specification fixed prior to any forecast
    print(pd.DataFrame(drift_tests).T.assign(Drift=drifts).round(3).to_string(), "\n")

    forecasts = {"EWMA": ewma_quantiles(s2), "EWMA-FHS": fhs_quantiles(dx_s, s2)}
    pits = {"EWMA": pd.Series(ndtr(dx_s / np.sqrt(s2)), index=df.index), "EWMA-FHS": fhs_pit(dx_s, s2)}
    p_stress = {}
    for name, (a, student) in ms_specs.items():
        forecasts[name], extra = ms_ou_forecast(df, a, student, drifts[name])
        pits[name], p_stress[name] = extra["pit"], extra["p_stress"]
    RESULTS.mkdir(exist_ok=True)
    for name, q in forecasts.items():
        assert q.loc[evaluable].notna().all().all()
        q.loc[evaluable].to_csv(RESULTS / f"forecast_{FORECASTS[name]}.csv", float_format="%.6f")

    eps_q = (forecasts["EWMA-FHS"].loc[evaluable] / np.sqrt(s2.loc[evaluable]).values[:, None]).mean()
    print(pd.DataFrame({"Normal": [ndtri(p) for p in LEVELS.values()], "FHS mean": eps_q.values}, index=list(LEVELS)).round(3), "\n")
    years = df.index.year
    print(pd.DataFrame({"sd dX": dx_s.groupby(years).std(), "sd dX / L": (dx_s / level).groupby(years).std()}).T.round(3).to_string(), "\n")

    fits = {name: (fit_ms_ou(x_lag, dx, a, student, drifts[name]), a, drifts[name]) for name, (a, student) in ms_specs.items()}
    for name, (p, _, _) in fits.items():
        print(name, "\n", ms_ou_table(p).round(3).to_string(), "\n")
    print(information_criteria(x_lag, dx, s2.to_numpy(), fits).round(1).to_string(), "\n")

    pits = {k: u.loc[evaluable] for k, u in pits.items()}
    in_sample = {f"{name}, in-sample": pd.Series(in_sample_pit(p, x_lag, dx, a), index=df.index).loc[evaluable]
                 for name, (p, a, _) in fits.items()}
    table = calibration(pits | in_sample)
    print(table.round(3).to_string(), "\n")

    periods = {"2019": ("2019-01-01", "2019-12-31"), "2022": ("2022-01-01", "2022-12-31")}
    mean_q = pd.DataFrame({(name, per): q.loc[evaluable].loc[a:b, ["q01", "q99"]].mean()
                           for name, q in forecasts.items() for per, (a, b) in periods.items()}).T
    print(mean_q.round(3).to_string(), "\n")
    print(pd.DataFrame({name: (p > 0.5).groupby(p.index.year).mean() for name, p in p_stress.items()}).T.round(2).to_string())

    plot_regimes(df, p_stress["MS-OU constant vol."], p_stress["MS-OU level vol."])
    plot_pit(pits, table)
