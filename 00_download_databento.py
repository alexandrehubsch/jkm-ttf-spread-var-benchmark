"""
Phase 0 — One-off download of daily settlement prices for front TTF and JKM contracts from Databento,
and of the ECB EUR/USD reference rate (free public API, no key required).

    python 00_download_databento.py --dry-run   # cost estimate only (free)
    python 00_download_databento.py             # estimate, then download under cap

CME futures (GLBX.MDP3 dataset), settlement prices from statistics schema, contracts identified by maturity rank
(symbology ".c.k", k = 0 for front-month). CME TTF settlement matches ICE Endex within CME tick size
(<= 0.0025 EUR/MWh, verified on ICE samples from 2019, 2022, and 2024); ICE settlements cost ~500 USD on
Databento, ICE close (ohlcv-1d) is last trade starting 2018-12-23, and ICE JKM is not published.
JKM trades minimally on Globex: only settlement exists, the same format applies to both legs.
Key read from DATABENTO_API_KEY or `.env`. Raw files are stored in `data/raw/`: any existing CSV or chunk
is never re-requested, hence never re-billed.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import databento as db
import pandas as pd

from common import DATA, ROOT

RAW = DATA / "raw"
DATASET, START = "GLBX.MDP3", "2018-01-01"
MAX_COST_USD = 3.0  # continuous contract cap (billed in USD; 3 USD < 3 EUR)
RETRIES = 3
LEGS = {"ttf": ("TTF", 3, "TTF_databento.csv"),  # Dutch TTF, EUR/MWh: front, 2nd, and 3rd contracts
        "jkm": ("JKM", 2, "JKM_databento.csv")}  # JKM (Platts), USD/MMBtu: front and 2nd contracts
MONTH_CODES = "FGHJKMNQUVXZ"
ECB_URL = f"https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata&startPeriod={START}"
ECB_CSV = "EUR_USD_ECB.csv"


def ecb_eurusd():
    """ECB daily reference rate (USD per EUR), one row per publication day."""
    fx = pd.read_csv(ECB_URL, usecols=["TIME_PERIOD", "OBS_VALUE"])
    fx.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "EURUSD"}).to_csv(DATA / ECB_CSV, index=False)
    print(f"{ECB_CSV}: {len(fx)} days, {fx['TIME_PERIOD'].iloc[0]} -> {fx['TIME_PERIOD'].iloc[-1]}")


def api_key():
    env = dict(line.strip().split("=", 1) for line in open(ROOT / ".env") if "=" in line) if (ROOT / ".env").exists() else {}
    return os.environ.get("DATABENTO_API_KEY") or env.get("DATABENTO_API_KEY") or sys.exit("DATABENTO_API_KEY not found")


def retry(call):
    """Repeated API call in case of transient error (504, dropped stream)."""
    for attempt in range(RETRIES):
        try:
            return call()
        except db.BentoError:
            if attempt == RETRIES - 1:
                raise


def yearly_requests(symbol, end):
    """One request per calendar year up to `end`."""
    start = pd.Timestamp(START, tz="UTC")
    bounds = [start, *pd.date_range(start, end, freq="YS")[1:], end]
    return [dict(dataset=DATASET, symbols=[symbol], stype_in="continuous", schema="statistics", start=a, end=b)
            for a, b in zip(bounds[:-1], bounds[1:])]


def fetch(key, path, req):
    """Writes to a `.part` file, renamed once stream is complete; retries if stream drops."""
    part = path.with_name(path.name + ".part")
    retry(lambda: (part.unlink(missing_ok=True), db.Historical(key).timeseries.get_range(**req, path=part)))
    part.replace(path)


def settlements(files):
    """One settlement per trading session, for the contract of that rank on that day, preferably final."""
    df = pd.concat([db.DBNStore.from_file(f).to_df() for f in files])
    df = df[df["stat_type"] == db.StatType.SETTLEMENT_PRICE].sort_values("ts_recv")
    df["Date"] = df["ts_ref"].dt.tz_convert(None).dt.normalize()  # trading session to which settlement refers
    df = df[df["instrument_id"] == df.groupby("Date")["instrument_id"].transform("first")]  # excludes late settlement of next contract
    df["final"] = df["stat_flags"] & 1  # bit 0: final settlement (otherwise preliminary)
    df = df.sort_values(["final", "ts_recv"]).drop_duplicates("Date", keep="last")
    return df.set_index("Date").sort_index()[["instrument_id", "price"]]


def delivery_months(client, df):
    """CME code of each contract (e.g. TTFV6) and delivery month; since instrument IDs can be reassigned,
    the mapping is dated."""
    ids = [str(i) for i in df["instrument_id"].unique()]
    res = retry(lambda: client.symbology.resolve(dataset=DATASET, symbols=ids, stype_in="instrument_id", stype_out="raw_symbol",
                                                 start_date=df.index.min().date(), end_date=(df.index.max() + pd.Timedelta(days=1)).date()))
    spans = pd.DataFrame([(int(i), pd.Timestamp(s["d0"]), pd.Timestamp(s["d1"]), s["s"])
                          for i, v in res["result"].items() for s in v], columns=["instrument_id", "d0", "d1", "symbol"])
    out = df.reset_index().merge(spans, on="instrument_id")
    out = out[(out["Date"] >= out["d0"]) & (out["Date"] < out["d1"])]
    code = out["symbol"].str.extract(r"([FGHJKMNQUVXZ])(\d)$")  # month and last digit of year
    year = out["Date"].dt.year + (code[1].astype(int) - out["Date"].dt.year) % 10
    out["delivery"] = pd.PeriodIndex.from_fields(year=year, month=code[0].map(MONTH_CODES.index) + 1, freq="M")
    assert len(out) == len(df) and (out["delivery"] > out["Date"].dt.to_period("M")).all()
    return out


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    DATA.mkdir(exist_ok=True)
    if not (DATA / ECB_CSV).exists() and not dry_run:
        ecb_eurusd()
    key = api_key()
    client = db.Historical(key)
    RAW.mkdir(parents=True, exist_ok=True)
    end = min(pd.Timestamp(retry(lambda: client.metadata.get_dataset_range(dataset=DATASET))["schema"]["statistics"]["end"]),
              pd.Timestamp.now(tz="UTC").normalize())  # complete available days
    for leg, (root, n, csv) in LEGS.items():
        if (DATA / csv).exists():
            print(f"{csv} already present")
            continue
        chunks = {k: {RAW / f"{leg}{k}_{r['start'].year}.dbn.zst": r for r in yearly_requests(f"{root}.c.{k}", end)}
                  for k in range(n)}
        todo = {}
        for k, files in chunks.items():
            missing = {f: r for f, r in files.items() if not f.exists()}
            if missing:
                reqs = list(missing.values())
                cost = retry(lambda: client.metadata.get_cost(**{**reqs[0], "end": reqs[-1]["end"]}))  # free call
                print(f"{root}.c.{k}: {len(missing)} years to download, estimated cost {cost:.4f} USD")
                if cost > MAX_COST_USD:
                    sys.exit(f"{root}.c.{k}: cost exceeds cap of {MAX_COST_USD} USD")
                todo |= missing
        if dry_run:
            continue
        if todo:
            with ThreadPoolExecutor(min(len(todo), 12)) as pool:  # server filters ~1-2 min per year: parallel chunks
                list(pool.map(lambda fr: fetch(key, *fr), todo.items()))
        df = pd.concat([settlements(files).assign(position=k) for k, files in chunks.items()])
        df = delivery_months(client, df).sort_values(["Date", "position"])
        assert not df.duplicated(["Date", "position"]).any()
        assert (df.groupby("Date")["delivery"].diff().dropna().map(lambda d: d.n) == 1).all()  # consecutive maturities
        df.rename(columns={"price": "settlement"})[["Date", "position", "symbol", "delivery", "settlement"]] \
            .to_csv(DATA / csv, index=False, float_format="%.6f")
        print(f"{csv}: {df['Date'].nunique()} sessions, {df['symbol'].nunique()} contracts, "
              f"{df['Date'].min():%Y-%m-%d} -> {df['Date'].max():%Y-%m-%d}")
