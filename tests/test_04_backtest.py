import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from pytest import approx
from scipy import stats

from common import DATA, FORECASTS, RESULTS, load_forecast, load_spread


@pytest.mark.parametrize("n, x, alpha", [(1867, 40, 0.01), (1867, 20, 0.01), (328, 0, 0.05)])
def test_kupiec_is_binomial_likelihood_ratio(phase, n, x, alpha):
    hit = pd.Series(np.arange(n) < x)
    lr, p = phase(4).kupiec(hit, alpha)
    assert lr == approx(2 * (stats.binom.logpmf(x, n, x / n) - stats.binom.logpmf(x, n, alpha)))
    assert p == approx(stats.chi2.sf(lr, 1))


def test_christoffersen_independence(phase):
    independent = np.zeros(401, bool)
    independent[[10, 11]] = True
    independent[30:390:20] = True  # pi_01 = pi_11 = 1/20 : LR_ind = 0
    clustered = np.zeros(401, bool)
    clustered[10:390:20] = clustered[11:390:20] = True  # paired breaches: pi_11 = 1/2
    assert phase(4).christoffersen(pd.Series(independent), 0.05)[0] == approx(1.0)
    assert phase(4).christoffersen(pd.Series(clustered), 0.1)[0] < 1e-6


def test_tick_loss_is_minimised_by_true_quantile(phase, rng):
    y = pd.Series(rng.standard_normal(200_000))
    loss = {s: phase(4).tick_loss(y, stats.norm.ppf(0.01) + s, 0.01).mean() for s in (-0.3, 0.0, 0.3)}
    assert loss[0.0] < min(loss[-0.3], loss[0.3])


def test_diebold_mariano_matches_statsmodels_hac(phase, rng):
    d = pd.Series(rng.standard_normal(1867)).rolling(3, min_periods=1).mean() + 0.05
    lags = int(4 * (len(d) / 100) ** (2 / 9))
    ref = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    stat, p = phase(4).diebold_mariano(d)
    assert (stat, p) == approx((ref.tvalues.iloc[0], ref.pvalues.iloc[0]), rel=1e-8)


def test_saved_coverage_table_is_reproduced(phase):
    if not ((RESULTS / "backtest_coverage.csv").exists() and (DATA / "spread.csv").exists()):
        pytest.skip("data/ or results/ absent: run scripts 00 to 04")
    forecasts = {name: load_forecast(name) for name in FORECASTS}
    dx = load_spread()["dX"].loc[forecasts["EWMA"].index]
    saved = pd.read_csv(RESULTS / "backtest_coverage.csv", index_col=[0, 1]).loc["Full"].reset_index(drop=True)
    pd.testing.assert_frame_equal(phase(4).coverage(dx, forecasts), saved, check_dtype=False, rtol=0, atol=1e-6)
