import numpy as np
import pandas as pd
import pytest
from pytest import approx
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def test_ewma_variance_is_frozen_after_missing_day(phase):
    dx = pd.Series([0.1, -0.2, 0.3, np.nan, 0.5], index=pd.bdate_range("2018-12-26", periods=5))
    s2 = phase(3).ewma_variance(dx, lam=0.9).to_numpy()
    assert s2[0] == approx(dx.loc["2018"].var())
    assert s2[1] == approx(0.9 * s2[0] + 0.1 * 0.1**2)
    assert s2[4] == s2[3]


def test_fhs_quantile_uses_only_past_residuals(phase, rng):
    p3 = phase(3)
    dx = pd.Series(rng.standard_t(4, 400), index=pd.bdate_range("2018-01-01", periods=400))
    dx.iloc[::30] = np.nan
    s2 = p3.ewma_variance(dx)
    q = p3.fhs_quantiles(dx, s2)["q01"]
    t = 350
    eps = (dx / np.sqrt(s2)).iloc[t - 250:t].dropna()
    assert q.iloc[t] == approx(np.quantile(eps, 0.01) * np.sqrt(s2.iloc[t]))
    shocked = dx.copy()
    shocked.iloc[t] = 100.0
    assert p3.fhs_quantiles(shocked, p3.ewma_variance(shocked))["q01"].iloc[t] == q.iloc[t]


def test_standard_densities(phase):
    p3, z = phase(3), np.linspace(-6, 6, 25)
    assert np.allclose(p3.std_pdf(z, None), stats.norm.pdf(z)) and np.allclose(p3.std_cdf(z, None), stats.norm.cdf(z))
    assert np.allclose(p3.std_pdf(z, 5.8), stats.t.pdf(z, 5.8)) and np.allclose(p3.std_cdf(z, 5.8), stats.t.cdf(z, 5.8))


def test_hamilton_filter_matches_statsmodels(phase, simulate_ou):
    """Constant-volatility MS-OU = Markov regression of dX_t on X_{t-1}, regime-dependent constant and variance."""
    params = np.array([0.05, 0.2, 0.5, -1.0, 0.3, 1.5, 0.95, 0.9])
    x = simulate_ou(600, theta=0.05, mu=0.5, sigma=0.5)
    x_lag, dx = x[:-1], np.diff(x)
    theta, mu, sigma, (p11, p22) = params[0:2], params[2:4], params[4:6], params[6:8]
    b = 1 - np.exp(-theta)
    c2 = sigma**2 * (1 - np.exp(-2 * theta)) / (2 * theta)
    model = MarkovRegression(dx, k_regimes=2, exog=x_lag, switching_variance=True)
    assert model.param_names == ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "x1[0]", "x1[1]", "sigma2[0]", "sigma2[1]"]
    reference = model.loglike(np.r_[p11, 1 - p22, b * mu, -b, c2])
    assert phase(3).hamilton(params, x_lag, dx)[1] == approx(reference, rel=1e-10)


@pytest.mark.parametrize("nu", [None, 5.0])
def test_mixture_quantile_inverts_mixture_cdf(phase, nu):
    p3 = phase(3)
    w, m, c = np.array([0.7, 0.3]), np.array([0.1, -0.5]), np.array([1.0, 3.0])
    for p in (0.01, 0.05, 0.95, 0.99):
        q = p3.mixture_quantile(p, w, m, c, nu)
        assert w @ p3.std_cdf((q - m) / c, nu) == approx(p, abs=1e-10)
    assert p3.mixture_quantile(0.01, np.array([1.0, 0.0]), m, c, None) == approx(m[0] + c[0] * stats.norm.ppf(0.01))


def test_fit_ms_ou_recovers_simulated_regimes(phase, rng, simulate_ou):
    n, p_stay = 2500, np.array([0.99, 0.97])
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay[s[t - 1]] else 1 - s[t - 1]
    x = simulate_ou(n, theta=[0.03, 0.1], mu=[1.0, -3.0], sigma=[0.2, 2.0], regimes=s)
    p = phase(3).fit_ms_ou(x[:-1], np.diff(x))
    assert p[4:6] == approx([0.2, 2.0], rel=0.15)
    assert p[6:8] == approx(p_stay, abs=0.02)
    assert p[3] == approx(-3.0, abs=1.5)


def test_drift_selection_and_fit_recover_regime_1_reversion(phase, rng):
    """OU drift in regime 1, random walk in regime 2, scale a_t = 1."""
    n, p_stay, theta, mu, k = 3000, np.array([0.98, 0.9]), 0.05, 1.0, np.array([0.3, 1.5])
    s, x = np.zeros(n, int), np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay[s[t - 1]] else 1 - s[t - 1]
        drift = (1 - np.exp(-theta)) * (mu - x[t - 1]) if s[t] == 0 else 0.0
        x[t] = x[t - 1] + drift + k[s[t]] * rng.standard_normal()
    p = phase(3).fit_ms_ou(x[:-1], np.diff(x), np.ones(n - 1), drift="calm")
    assert phase(3).select_drift(x[:-1], np.diff(x), np.ones(n - 1))[0] == "calm"
    assert p[1] == p[3] == 0
    assert p[0] == approx(theta, rel=0.5) and p[4:6] == approx(k, rel=0.1) and p[6:8] == approx(p_stay, abs=0.03)
