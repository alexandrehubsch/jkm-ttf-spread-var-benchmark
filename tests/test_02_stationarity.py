import numpy as np
import pandas as pd
from pytest import approx
from statsmodels.tsa.stattools import adfuller


def frame(x):
    df = pd.DataFrame({"spread": x}, index=pd.bdate_range("2018-01-01", periods=len(x)))
    df["dX"] = df["spread"].diff()
    return df


def test_adf_matches_statsmodels_without_missing_days(phase, simulate_ou):
    x = simulate_ou(800, theta=0.05, mu=1.0, sigma=1.0)
    t, p, lags = phase(2).adf(frame(x))
    ref = adfuller(x, maxlag=int(12 * (len(x) / 100) ** 0.25), regression="c", autolag="BIC", result_object=False)
    assert lags == ref[2]
    assert (t, p) == approx((ref[0], ref[1]), rel=1e-8)


def test_fit_ou_recovers_parameters_with_missing_days(phase, simulate_ou):
    df = frame(simulate_ou(20_000, theta=0.05, mu=1.0, sigma=0.8))
    df.loc[df.index[::21], "dX"] = np.nan
    est = phase(2).fit_ou(df)
    assert est["theta"] == approx(0.05, rel=0.15)
    assert est["mu"] == approx(1.0, abs=0.3)
    assert est["sigma"] == approx(0.8, rel=0.03)


def test_variance_ratio_detects_mean_reversion_but_not_random_walk(phase, rng, simulate_ou):
    vr = phase(2).variance_ratio
    noise = pd.Series(rng.standard_normal(5000))
    v, p = vr(noise, 20)
    assert v == approx(1, abs=0.1) and p > 0.01
    rho = [noise.autocorr(j) for j in range(1, 20)]
    assert v == approx(1 + 2 * sum((1 - j / 20) * r for j, r in enumerate(rho, 1)), abs=0.02)  # VR(q) = 1 + 2 sum (1 - j/q) rho_j
    v, p = vr(pd.Series(simulate_ou(5000, theta=0.1, mu=0.0, sigma=1.0)).diff(), 20)
    assert v < 0.6 and p < 0.01
