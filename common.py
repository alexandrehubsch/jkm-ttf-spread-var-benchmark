"""
Shared elements across phase scripts: paths, figure styles, reference periods, data loading.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
DATA, FIG, RESULTS = ROOT / "data", ROOT / "figures", ROOT / "results"
END = "2026-10-07"  # last session of the published study; later downloads are truncated for reproducibility

PERIODS = {
    "Calm market (2019)": ("2019-01-01", "2019-12-31"),
    "JKM squeeze (winter 2020-21)": ("2020-11-01", "2021-03-31"),
    "European energy crisis (2022)": ("2022-01-01", "2022-12-31"),
    "Recent period (2026)": ("2026-01-01", None),
}
STRESS = ("2021-09-01", "2022-12-31")  # autumn 2021 European price surge and 2022 crisis

FORECASTS = {"EWMA": "ewma", "EWMA-FHS": "fhs", "MS-OU constant vol.": "ms_ou_const",
             "MS-OU level vol.": "ms_ou_level", "MS-OU EWMA vol.": "ms_ou_ewma"}

COLORS = {"dark": "#2b3a4a", "blue": "#4a7298", "red": "#b0594b", "light": "#cfdbe6", "light_red": "#efdcd6", "sand": "#ede5d6"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 9, "axes.titlesize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 0.7, "axes.prop_cycle": plt.cycler(color=[COLORS["blue"], COLORS["red"], COLORS["dark"]]),
})


def load_spread():
    return pd.read_csv(DATA / "spread.csv", index_col="Date", parse_dates=True)


def load_forecast(name):
    return pd.read_csv(RESULTS / f"forecast_{FORECASTS[name]}.csv", index_col="Date", parse_dates=True)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
