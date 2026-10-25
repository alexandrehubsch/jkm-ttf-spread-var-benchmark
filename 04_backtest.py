"""
Phase 4 — One-day VaR backtesting: coverage (Kupiec, Christoffersen), quantile loss, and Diebold–Mariano tests.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from common import COLORS, FIG, FORECASTS, RESULTS, STRESS, load_forecast, load_spread, save

OUT = FIG / "04_backtest"
CASES = {"1% long": ("q01", 0.01), "1% short": ("q99", 0.99), "5% long": ("q05", 0.05), "5% short": ("q95", 0.95)}
TICKS = [-10, -1, 0, 1, 10]
LINEAR_WIDTH = 0.2  # asinh scale: linear for |dX| < 0.2, logarithmic beyond
CALM = ("2024-07-01", "2026-06-30")  # post-crisis showcase window, 24 months
PAIRS = [(m, "EWMA") for m in list(FORECASTS)[1:]] + [(m, "EWMA-FHS") for m in ["MS-OU level vol.", "MS-OU EWMA vol."]] \
    + [("MS-OU EWMA vol.", "MS-OU level vol.")]


def hits(dx, q, p):
    """VaR breach: dX_t < q_t(p) for long positions (p < 0.5), dX_t > q_t(p) for short positions."""
    return (dx < q) if p < 0.5 else (dx > q)


def kupiec(hit, alpha):
    """Unconditional coverage: LR_uc ~ chi2(1) under H0: P(I_t = 1) = alpha."""
    n, x = len(hit), hit.sum()
    pi = x / n
    lr = -2 * (xlogy(n - x, 1 - alpha) + xlogy(x, alpha) - xlogy(n - x, 1 - pi) - xlogy(x, pi))  # xlogy(0, 0) = 0
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(hit, alpha):
    """Independence (first-order Markov chain) and conditional coverage LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    h = hit.astype(int).to_numpy()
    prev, cur = h[:-1], h[1:]
    n00, n01 = np.sum((prev == 0) & (cur == 0)), np.sum((prev == 0) & (cur == 1))
    n10, n11 = np.sum((prev == 1) & (cur == 0)), np.sum((prev == 1) & (cur == 1))
    p01, p11, p = n01 / (n00 + n01), n11 / max(n10 + n11, 1), (n01 + n11) / len(cur)
    ll_markov = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll_iid = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    lr_ind = -2 * (ll_iid - ll_markov)
    lr_cc = kupiec(hit, alpha)[0] + lr_ind
    return stats.chi2.sf(lr_ind, 1), stats.chi2.sf(lr_cc, 2)


def tick_loss(dx, q, p):
    return (p - (dx < q)) * (dx - q)


def diebold_mariano(d):
    """DM = mean(d) / sqrt(long-run variance / n), Newey–West variance (Bartlett kernel)."""
    d = np.asarray(d)
    n = len(d)
    lags = int(4 * (n / 100) ** (2 / 9))  # Newey and West (1994) rule
    u = d - d.mean()
    lrv = u @ u / n + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n for k in range(1, lags + 1))
    stat = d.mean() / np.sqrt(lrv / n)
    return stat, 2 * stats.norm.sf(abs(stat))


def coverage(dx, forecasts):
    rows = []
    for model, q in forecasts.items():
        for case, (col, p) in CASES.items():
            alpha = min(p, 1 - p)
            hit = hits(dx, q[col], p)
            rows.append({"Model": model, "Case": case, "N": len(hit), "Breaches": hit.sum(), "Rate": hit.mean(),
                         "Kupiec p": kupiec(hit, alpha)[1], **dict(zip(["Ind. p", "CC p"], christoffersen(hit, alpha))),
                         "Tick loss": tick_loss(dx, q[col], p).mean()})
    return pd.DataFrame(rows)


def dm_table(dx, forecasts):
    rows = []
    for a, b in PAIRS:
        for case, (col, p) in CASES.items():
            d = tick_loss(dx, forecasts[a][col], p) - tick_loss(dx, forecasts[b][col], p)
            stat, pval = diebold_mariano(d)
            rows.append({"A": a, "B": b, "Case": case, "Mean diff": d.mean(), "DM": stat, "p": pval})
    return pd.DataFrame(rows)


def plot_breaches(df, dx, forecasts, start, end, name, spread=True, asinh=False):
    """1% and 99% quantiles and breaches; asinh scale so calm periods remain legible alongside crisis."""
    rows = len(forecasts) + spread
    fig, axes = plt.subplots(rows, 1, figsize=(6.5, 1.45 * rows + 0.4), sharex=True)
    if spread:
        axes[0].plot(df["spread"].loc[start:end], color=COLORS["dark"])
        axes[0].axvspan(*map(pd.Timestamp, STRESS), color=COLORS["sand"], lw=0, zorder=0)
        axes[0].set(ylabel=r"$X_t$", title="Spread (stress window shaded)")
    y = dx.loc[start:end]
    for ax, (model, q) in zip(axes[spread:], forecasts.items()):
        q = q.loc[start:end]
        low, high = y < q["q01"], y > q["q99"]
        ax.fill_between(q.index, q["q01"], q["q99"], color=COLORS["light"], lw=0, label="1% to 99% forecast interval")
        ax.plot(y, color=COLORS["dark"], lw=0.3)
        ax.plot(y[low], "o", color=COLORS["red"], ms=2.2, label=f"long breaches: {low.sum()} (expected {0.01 * len(y):.1f})")
        ax.plot(y[high], "o", color=COLORS["blue"], mfc="white", ms=2.2, label=f"short breaches: {high.sum()} (expected {0.01 * len(y):.1f})")
        ax.set(ylabel=r"$\Delta X_t$", title=model)
        ax.legend(loc="upper left", ncol=3, fontsize=6.5, frameon=True, facecolor="white", edgecolor="0.7", framealpha=1)
    lo = min(y.min(), *(q["q01"].loc[start:end].min() for q in forecasts.values()))
    hi = max(y.max(), *(q["q99"].loc[start:end].max() for q in forecasts.values()))
    to, back = (lambda v: np.arcsinh(v / LINEAR_WIDTH), lambda u: LINEAR_WIDTH * np.sinh(u)) if asinh else (lambda v: v, lambda u: u)
    a, b = to(lo), to(hi)
    for ax in axes[spread:]:
        if asinh:
            ax.set_yscale("asinh", linear_width=LINEAR_WIDTH)
            ax.set_yticks(TICKS, [f"${v:g}$" for v in TICKS])
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.set_ylabel(r"$\Delta X_t$ (asinh scale)")
        ax.set_ylim(back(a - 0.05 * (b - a)), back(b + 0.35 * (b - a)))  # margins computed in display scale
    axes[-1].set_xlim(pd.Timestamp(y.index[0]), pd.Timestamp(y.index[-1]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.tight_layout()
    save(fig, OUT / f"{name}.png")


def plot_cumulative(dx, forecasts):
    """Cumulative 1% breaches per direction, expected count, and 95% binomial interval under H0."""
    t = np.arange(1, len(dx) + 1)
    lo, hi = stats.binom.ppf(0.025, t, 0.01), stats.binom.ppf(0.975, t, 0.01)  # pointwise interval, date by date
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6), sharex=True, sharey=True)
    for ax, (model, q) in zip(axes.flat, forecasts.items()):
        ax.axvspan(*map(pd.Timestamp, STRESS), color=COLORS["sand"], lw=0, zorder=0)
        ax.fill_between(dx.index, lo, hi, color=COLORS["light"], lw=0, label="95% interval under $H_0$")
        ax.plot(dx.index, 0.01 * t, color="0.4", ls=":", label="expected (1%)")
        ax.plot(hits(dx, q["q01"], 0.01).cumsum(), color=COLORS["red"], label="long breaches")
        ax.plot(hits(dx, q["q99"], 0.99).cumsum(), color=COLORS["blue"], ls="--", label="short breaches")
        ax.set_title(model)
    axes[-1, -1].remove()
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative breaches")
    for ax in axes[-1, :1].tolist() + [axes[1, 1]]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelbottom=True)
    axes[0, 0].set_xlim(dx.index[0], dx.index[-1])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.75, 0.18), fontsize=8,
               title="1% VaR, stress window shaded", title_fontsize=8)
    fig.tight_layout()
    save(fig, OUT / "cumulative_breaches.png")


def plot_comparison(cov, dm):
    """Full-sample summary: (a) breach rate against Kupiec acceptance region, (b) quantile loss relative to EWMA."""
    models = list(FORECASTS)
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(2, len(CASES), figsize=(6.5, 4.2), sharey=True)
    kupiec_bg = "#e5e9ec"
    for j, (case, (_, p)) in enumerate(CASES.items()):
        alpha, c = min(p, 1 - p), cov[cov.Case == case].set_index("Model").loc[models]
        n = c["N"].iloc[0]
        accepted = [x for x in range(int(3 * alpha * n)) if kupiec(pd.Series(np.arange(n) < x), alpha)[1] >= 0.05]
        rejected = (c[["Kupiec p", "Ind. p", "CC p"]] < 0.05).any(axis=1)
        ax = axes[0, j]
        ax.axvspan(100 * accepted[0] / n, 100 * accepted[-1] / n, color=kupiec_bg, lw=0)
        ax.axvline(100 * alpha, color="0.4", ls=":")
        ax.scatter(100 * c["Rate"], y, s=24, c=np.where(rejected, COLORS["red"], "white"), edgecolors=np.where(rejected, COLORS["red"], COLORS["dark"]), lw=0.9, zorder=3)
        ax.set_title(case)
        rel = 100 * (c["Tick loss"] / c.loc["EWMA", "Tick loss"] - 1)
        significant = dm[(dm.Case == case) & (dm.B == "EWMA")].set_index("A")["p"].reindex(models) < 0.05
        ax = axes[1, j]
        ax.barh(y, rel, height=0.6, color=np.where(significant, COLORS["blue"], COLORS["light"]))
        ax.axvline(0, color="k", lw=0.6)
        for yi, r, m in zip(y, rel, models):
            ax.text(r + (2 if r >= 0 else -2), yi, "ref." if m == "EWMA" else f"${r:+.0f}$", va="center", fontsize=6.5,
                    ha="left" if r >= 0 else "right")
        span = max(rel.max(), 0) - min(rel.min(), 0)
        ax.set_xlim(min(rel.min(), 0) - 0.3 * span, max(rel.max(), 0) + 0.3 * span)
    axes[0, 0].set_yticks(y, models)
    axes[0, 0].set_ylim(y.min() - 0.6, y.max() + 0.6)
    fig.text(0.01, 0.99, "(a) Breach rate (%) and Kupiec acceptance region", fontsize=9, va="top")
    fig.text(0.01, 0.53, "(b) Mean tick loss relative to EWMA (%)", fontsize=9, va="top")
    handles = [Patch(color=kupiec_bg, label="Kupiec acceptance region (5%)"), Line2D([], [], color="0.4", ls=":", label="nominal rate"),
               Line2D([], [], ls="", marker="o", mfc=COLORS["red"], mec=COLORS["red"], label="rejected by Kupiec or Christoffersen (5%)"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=COLORS["dark"], label="not rejected"),
               Patch(color=COLORS["blue"], label="Diebold-Mariano significant (5%)"), Patch(color=COLORS["light"], label="not significant")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96), h_pad=2.5)
    save(fig, OUT / "comparison.png")


if __name__ == "__main__":
    df = load_spread()
    forecasts = {name: load_forecast(name) for name in FORECASTS}
    index = forecasts["EWMA"].index
    assert all(q.index.equals(index) for q in forecasts.values())
    dx = df["dX"].loc[index]
    assert dx.notna().all() and len(dx) == 1867

    stress = (index >= STRESS[0]) & (index <= STRESS[1])
    samples = {"Full": slice(None), "Stress": stress, "Non-stress": ~stress}
    cov = pd.concat({k: coverage(dx[m], {n: q[m] for n, q in forecasts.items()}) for k, m in samples.items()}, names=["Sample"])
    dm = pd.concat({k: dm_table(dx[m], {n: q[m] for n, q in forecasts.items()}) for k, m in samples.items()}, names=["Sample"])
    RESULTS.mkdir(exist_ok=True)
    cov.to_csv(RESULTS / "backtest_coverage.csv", float_format="%.6f")
    dm.to_csv(RESULTS / "backtest_dm.csv", float_format="%.6f")

    pd.set_option("display.width", 200)
    for sample in samples:
        print(f"\n{sample} sample : coverage and tick loss\n",
              cov.loc[sample].set_index(["Case", "Model"]).sort_index().round(4).to_string())
        print(f"\n{sample} sample : Diebold-Mariano on tick loss (A - B)\n",
              dm.loc[sample].set_index(["Case", "A", "B"]).sort_index().round(4).to_string())

    adaptive = {m: forecasts[m] for m in ["EWMA", "EWMA-FHS"]}
    plot_breaches(df, dx, adaptive, index[0], index[-1], "breaches_full_adaptive", asinh=True)
    plot_breaches(df, dx, {m: q for m, q in forecasts.items() if m not in adaptive}, index[0], index[-1], "breaches_full_ms_ou", asinh=True)
    plot_breaches(df, dx, forecasts, *CALM, "breaches_calm", spread=False)
    plot_breaches(df, dx, forecasts, *STRESS, "breaches_crisis", spread=False)
    plot_cumulative(dx, forecasts)
    plot_comparison(cov.loc["Full"], dm.loc["Full"])
