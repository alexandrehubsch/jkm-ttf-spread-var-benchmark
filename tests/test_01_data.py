import numpy as np
import pandas as pd
import pytest

from common import DATA, load_spread


def test_roll_day_is_first_common_day_on_new_contract(phase):
    contract = pd.Series([7, 7, 8, 8, 9], index=pd.bdate_range("2024-01-15", periods=5))
    assert phase(1).roll_days(contract).tolist() == [False, False, True, False, True]


def test_change_of_delivery_month_excludes_the_calendar_spread(phase):
    dates = pd.bdate_range("2024-01-15", periods=3)
    mar, apr = pd.Period("2024-03", "M"), pd.Period("2024-04", "M")
    jkm = pd.DataFrame({mar: [10.0, 11.0, 12.0], apr: [20.0, 21.0, 23.0]}, index=dates)
    ttf = pd.DataFrame({mar: [1.0, 1.0, 1.5], apr: [5.0, 5.0, 5.0]}, index=dates)
    x, dx = phase(1).same_month_spread(jkm, ttf, pd.Series([mar, mar, apr], index=dates))
    assert x.tolist() == [9.0, 10.0, 18.0]
    assert np.isnan(dx.iloc[0]) and dx.iloc[1:].tolist() == [1.0, 0.5]  # March held until transition date


@pytest.fixture(scope="module")
def built(phase):
    p1 = phase(1)
    if not all((DATA / f).exists() for f in p1.FILES.values()):
        pytest.skip("raw data absent from data/")
    return p1.build_spread()


def test_spread_csv_is_reproduced_from_raw_data(built):
    saved = load_spread()
    cols = saved.columns.drop("month")
    pd.testing.assert_frame_equal(built[cols], saved[cols], check_freq=False, rtol=0, atol=1e-6)


def test_roll_days_follow_expiry_calendars(built):
    jkm, ttf = built.index[built["roll_jkm"]], built.index[built["roll_ttf"]]
    assert jkm.day.min() >= 16 and jkm.day.max() <= 19  # expiration on the 15th (or preceding business day)
    assert ((ttf.day >= 26) | (ttf.day <= 2)).all()  # expiration two UK business days before the 1st


def test_no_day_removed_and_month_follows_jkm_pricing_period(built):
    assert built["dX"].iloc[1:].notna().all()
    assert built["switch"].equals(built["roll_jkm"])  # 2nd JKM maturity rolls when the 1st expires
    m = pd.PeriodIndex(built["month"])
    diff_months = (m.year - built.index.year) * 12 + (m.month - built.index.month)
    assert np.isin(diff_months, [2, 3]).all()
