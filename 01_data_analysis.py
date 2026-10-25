"""
Phase 1 — Construction of the JKM - TTF spread (USD/MMBtu) and descriptive analysis.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from common import COLORS, DATA, END, FIG, PERIODS, save

OUT = FIG / "01_data_analysis"
FILES = {
    "ttf": "TTF_databento.csv",   # CME Dutch TTF futures, settlements for the first three maturities, EUR/MWh
    "jkm": "JKM_databento.csv",   # CME JKM (Platts) futures, settlements for the first two maturities, USD/MMBtu
    "eurusd": "EUR_USD_ECB.csv",  # ECB reference exchange rate
}

CALM = ("2019-01-01", "2020-11-30")
MMBTU_PER_MWH = 3.412142


def load_curve(file):
    """Settlements by trading session and delivery month, and delivery month of each maturity rank (0 = front-month)."""
    df = pd.read_csv(DATA / file, parse_dates=["Date"])
    df["delivery"] = pd.PeriodIndex(df["delivery"], freq="M")
    return df.pivot(index="Date", columns="delivery", values="settlement"), df.pivot(index="Date", columns="position", values="delivery")


def pick(curve, months):
    """Price of each session for the indicated delivery month (NaN if month is not quoted)."""
    col = curve.columns.get_indexer(months)
    return pd.Series(np.where(col >= 0, curve.to_numpy()[np.arange(len(curve)), col], np.nan), index=curve.index)


def same_month_spread(jkm, ttf, month):
    """Spread for delivery month M_t and change dX_t of contracts held from t-1 to t (month M_{t-1}): on roll day,
    the price gap between contracts never enters dX_t."""
    held = month.shift(1)
    x = pick(jkm, month) - pick(ttf, month)
    return x, pick(jkm, held) - pick(ttf, held) - x.shift(1)


def stale(curve, month):
    """Session with no new quote: the held contract (month M_{t-1}) retains previous session settlement."""
    held = month.shift(1)
    return pick(curve, held).eq(pick(curve.shift(1), held))


def roll_days(contract):
    """First common day quoted on a new contract: the front-month series rolls to the next contract."""
    return contract.ne(contract.shift()) & contract.shift().notna()


def build_spread():
    """Spread on delivery month M_t of the second JKM maturity, the first outside Platts assessment window."""
    (ttf, ttf_rank), (jkm, jkm_rank) = load_curve(FILES["ttf"]), load_curve(FILES["jkm"])
    fx = pd.read_csv(DATA / FILES["eurusd"], index_col=0, parse_dates=True).iloc[:, -1].rename_axis("Date")  # FX rate in last column
    dates = ttf.index.intersection(jkm.index).intersection(fx.index)
    dates = dates[(dates <= END) & jkm_rank.loc[dates, 1].notna()]
    fresh = ~(stale(jkm.loc[dates], jkm_rank.loc[dates, 1]) | stale(ttf.loc[dates], jkm_rank.loc[dates, 1]))
    dates = dates[fresh]  # Platts holidays (UK, Singapore): JKM settlement carried over, change recorded next day
    fx, ttf, jkm, ttf_rank, jkm_rank = fx.loc[dates], ttf.loc[dates], jkm.loc[dates], ttf_rank.loc[dates], jkm_rank.loc[dates]
    ttf_usd = ttf.mul(fx, axis=0) / MMBTU_PER_MWH
    month = jkm_rank[1]
    df = pd.DataFrame({"month": month, "ttf_eur": pick(ttf, month), "eurusd": fx, "ttf": pick(ttf_usd, month), "jkm": pick(jkm, month)})
    df["spread"], df["dX"] = same_month_spread(jkm, ttf_usd, month)
    df["switch"] = roll_days(month)
    df["front"] = pick(jkm, jkm_rank[0]) - pick(ttf_usd, ttf_rank[0])  # front-month spread, to illustrate rolls
    df["front_jkm"], df["front_month"] = pick(jkm, jkm_rank[0]), jkm_rank[0]
    df["roll_jkm"], df["roll_ttf"] = roll_days(jkm_rank[0]), roll_days(ttf_rank[0])
    m = pd.PeriodIndex(month)
    assert np.isin((m[1:].year - m[:-1].year) * 12 + (m[1:].month - m[:-1].month), [0, 1]).all()  # one delivery month at a time, without gaps
    return df


def moments(x):
    x = x.dropna()
    return pd.Series({"N": len(x), "Mean": x.mean(), "Std": x.std(), "Skew": stats.skew(x),
                      "Ex. kurt": stats.kurtosis(x), "Min": x.min(), "Max": x.max()})


def plot_history(df):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6), sharex=True)
    axes[0].plot(df["ttf"], color=COLORS["blue"], label="TTF")
    axes[0].plot(df["jkm"], color=COLORS["red"], label="JKM")
    axes[0].set_ylabel("USD/MMBtu")
    axes[0].legend(loc="upper left")
    axes[1].plot(df["spread"], color=COLORS["dark"])
    axes[1].axhline(0, color="0.5", lw=0.5, ls="--")
    axes[1].set_ylabel(r"$X_t$")
    axes[2].plot(df["dX"], color=COLORS["dark"], lw=0.5)
    axes[2].set_ylabel(r"$\Delta X_t$")
    axes[2].set_xlim(df.index[0], df.index[-1])
    save(fig, OUT / "history.png")


def roll_stats(df):
    """Share of rolls in changes of a chained front-month spread, avoided by constant-contract construction."""
    raw = df["front"].diff()
    rows = {}
    for name, (start, end) in {"Full sample": (None, None), "Calm 2019-2020": CALM}.items():
        x, sub = raw.loc[start:end], df.loc[start:end]
        roll = sub["roll_jkm"] | sub["roll_ttf"]
        rows[name] = {"JKM roll days": sub["roll_jkm"].sum(), "TTF roll days": sub["roll_ttf"].sum(),
                      "Share JKM": (x[sub["roll_jkm"]] ** 2).sum() / (x ** 2).sum(),
                      "Share TTF": (x[sub["roll_ttf"]] ** 2).sum() / (x ** 2).sum(),
                      "Top-10 on roll": roll[x.abs().nlargest(10).index].sum()}
    return pd.DataFrame(rows).T


def stats_box(ax, x, extra=True):
    lines = [rf"$\sigma_{{d}}$ = {x.std():.2f}"]
    if extra:
        lines += [rf"$\sigma_{{a}}$ = {x.std() * np.sqrt(252):.2f}", f"Skew = {stats.skew(x):.2f}"]
    lines += [f"Ex. kurt = {stats.kurtosis(x):.1f}"]
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.7", "lw": 0.5})


def plot_roll(df):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.6))

    jkm = df.loc["2020-11-01":"2021-03-31", "front_jkm"]
    for i, (month, seg) in enumerate(jkm.groupby(df.loc[jkm.index, "front_month"])):
        if i:
            axes[0].plot([prev.index[-1], seg.index[0]], [prev.iloc[-1], seg.iloc[0]],
                         color="0.5", ls=":", label="Roll gap" if i == 1 else None)
        axes[0].plot(seg, color=COLORS["blue"], label="One contract" if i == 0 else None)
        axes[0].text(seg.index[len(seg) // 2], seg.max() + 1.2, month.strftime("%b %y"), ha="center", fontsize=7)
        prev = seg
    axes[0].set(ylabel="USD/MMBtu", ylim=(None, jkm.max() + 4),
                title="(a) The JKM front-month series is a chain of contracts (label = delivery month)")
    axes[0].legend(loc="upper left")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    sub = df.loc[CALM[0]:CALM[1]]
    raw = df["front"].diff().loc[CALM[0]:CALM[1]]
    axes[1].plot(raw, color=COLORS["dark"], lw=0.5)
    axes[1].plot(raw[sub["roll_jkm"]], "o", color=COLORS["red"], ms=2.5, label="JKM roll day")
    axes[1].plot(raw[sub["roll_ttf"]], "o", color=COLORS["blue"], mfc="white", ms=2.5, label="TTF roll day")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].set_title(r"(b) Chained front-month spread: raw daily changes, calm period 2019-2020")
    stats_box(axes[1], raw.dropna(), extra=False)
    axes[2].plot(sub["dX"], color=COLORS["dark"], lw=0.5)
    axes[2].set_title(r"(c) Same-contract $\Delta X_t$ and same delivery month for both legs")
    stats_box(axes[2], sub["dX"].dropna(), extra=False)
    for ax in axes[1:]:
        ax.set(ylabel="USD/MMBtu", ylim=(raw.min() * 1.15, raw.max() * 2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.tight_layout()
    save(fig, OUT / "roll.png")


def plot_zooms(df):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))
    for ax, (name, (start, end)) in zip(axes.flat, PERIODS.items()):
        sub = df.loc[start:end]
        dx = sub["dX"].dropna()
        ax.plot(sub["spread"], color=COLORS["dark"])
        ax.axhline(0, color="0.5", lw=0.5, ls="--")
        ax.set_title(name)
        lo, hi = sub["spread"].min(), sub["spread"].max()
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.6 * (hi - lo))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        stats_box(ax, dx)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$X_t$ (USD/MMBtu)")
    fig.tight_layout()
    save(fig, OUT / "zooms.png")


def plot_distributions(df):
    dx = df["dX"].dropna()
    z = (dx - dx.mean()) / dx.std()
    nu, loc, scale = stats.t.fit(z)
    grid = np.linspace(-8, 8, 400)
    dens, edges = np.histogram(z, bins=np.linspace(-8, 8, 81), density=True)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.8))
    ax = axes[0, 0]
    ax.stairs(dens, edges, fill=True, color=COLORS["light"])
    ax.plot(grid, stats.norm.pdf(grid), color=COLORS["dark"], label=r"$\mathcal{N}(0,1)$")
    ax.plot(grid, stats.t.pdf(grid, nu, loc, scale), color=COLORS["red"], ls="--", label=rf"Student-$t$ ($\nu$ = {nu:.1f})")
    ax.set(yscale="log", ylim=(1e-4, 3), xlabel=r"Standardised $\Delta X_t$", title="Density (log scale)")
    ax.legend(loc="upper left", fontsize=6.5)

    (osm, osr), _ = stats.probplot(z, dist="norm")
    axes[0, 1].plot(osm, osr, ".", color=COLORS["blue"], ms=1.5)
    axes[0, 1].plot([-4, 4], [-4, 4], color="0.5", lw=0.6)
    axes[0, 1].set(xlabel="Normal quantiles", ylabel="Empirical quantiles", title="Normal QQ-plot")

    gaps = df["dX"].iloc[1:]
    for ax, x, title in zip(axes[1], [gaps, gaps**2], [r"ACF of $\Delta X_t$", r"ACF of $(\Delta X_t)^2$"]):
        plot_acf(x, ax=ax, lags=40, zero=False, missing="conservative",
                 color=COLORS["blue"], vlines_kwargs={"colors": COLORS["blue"]}, markersize=2)
        ax.set(title=title, xlabel="Lag (days)", ylim=(-0.25, 0.6))
    fig.tight_layout()
    save(fig, OUT / "distributions.png")


if __name__ == "__main__":
    df = build_spread()
    raw = df["front"].diff()
    print(f"{len(df)} common days, {df['switch'].sum()} delivery month transitions, {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}\n")
    print(roll_stats(df).round(3).to_string(), "\n")
    table = pd.DataFrame({"X": moments(df["spread"]), "dX front raw": moments(raw), "dX": moments(df["dX"])}).T
    print(table.astype({"N": int}).round(3).to_string(), "\n")
    by_period = pd.DataFrame({name: moments(df["dX"].loc[a:b]) for name, (a, b) in PERIODS.items()}).T
    by_period["Std ann."] = by_period["Std"] * np.sqrt(252)
    print(by_period[["N", "Std", "Std ann.", "Skew", "Ex. kurt"]].astype({"N": int}).round(3).to_string())
    plot_history(df)
    plot_roll(df)
    plot_zooms(df)
    plot_distributions(df)
    cols = ["month", "ttf_eur", "eurusd", "ttf", "jkm", "spread", "switch", "dX"]
    DATA.mkdir(exist_ok=True)
    df[cols].to_csv(DATA / "spread.csv", float_format="%.6f")
