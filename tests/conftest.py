import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPTS = {1: "01_data_analysis", 2: "02_stationarity", 3: "03_var_models", 4: "04_backtest"}


@pytest.fixture(scope="session")
def phase():
    """phase(k) imports the script for phase k (name starts with a digit, hence importlib)."""
    return lambda k: importlib.import_module(SCRIPTS[k])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def simulate_ou(rng):
    """Exact discretization of an OU process (dt = 1 day); theta, mu, sigma may depend on a given regime s_t."""
    def simulate(n, theta, mu, sigma, regimes=None):
        s = np.zeros(n, int) if regimes is None else regimes
        theta, mu, sigma = (np.atleast_1d(v)[s] for v in (theta, mu, sigma))
        b, sd = np.exp(-theta), sigma * np.sqrt((1 - np.exp(-2 * theta)) / (2 * theta))
        x = np.empty(n)
        x[0] = mu[0]
        for t in range(1, n):
            x[t] = mu[t] + b[t] * (x[t - 1] - mu[t]) + sd[t] * rng.standard_normal()
        return x
    return simulate
