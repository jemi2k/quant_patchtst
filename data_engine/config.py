"""
data_engine/config.py
=====================
Static configuration for the Phase 2 data engine.

Holds everything the ingestion / bar-construction code needs that is NOT model
hyperparameters (those live in the top-level configs.py): the asset universe,
their approximate listing months, the Binance archive URLs, local storage paths,
and the aggTrades schema.

This module is import-only (no side effects), so the async downloader, the DIB
engine, and later phases can all import it without triggering I/O.
"""

# --- Asset universe ---------------------------------------------------------------
# Binance USDT-M perpetual futures, uppercase, one per liquid L1/Infra chain.
SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "LINKUSDT"]

# Approximate first available monthly archive file per symbol (YYYY-MM).
# Best-effort floors: the downloader skips 404s gracefully, so a wrong value only
# wastes a few HEAD requests and is corrected from the first-run log.
INCEPTION = {
    "ETHUSDT": "2019-11",
    "LINKUSDT": "2020-06",
    "SOLUSDT": "2021-04",
    "NEARUSDT": "2021-04",
    "AVAXUSDT": "2021-09",
    "APTUSDT": "2022-10",
    "SUIUSDT": "2023-05",
}

# --- Archive ----------------------------------------------------------------------
# data.binance.vision serves USDT-M futures monthly aggTrades as:
#   https://data.binance.vision/data/futures/um/monthly/aggTrades/{SYM}/{SYM}-aggTrades-{YYYY-MM}.zip
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades"

# --- Local storage (data/ is gitignored) ------------------------------------------
RAW_DIR = "data/raw/aggTrades"          # downloaded zips
PARQUET_DIR = "data/parquet/aggTrades"  # extracted + compressed (canonical store)
MANIFEST_PATH = "data/manifest.json"    # per-file row counts / sizes / status

# --- Zip retention -----------------------------------------------------------------
# df -h shows 316 GB free on / (the project partition). Rough estimate for the full
# 7-asset history is ~60-150 GB of raw zips + ~40-80 GB of parquet, so keeping both
# fits with headroom and gives an offline backup. If disk ever approaches ~80% usage
# mid-download, flip this to False and delete zips (they are re-downloadable from the
# archive, and parquet is the canonical store).
KEEP_RAW_ZIPS = True

# --- Concurrency & retries ---------------------------------------------------------
MAX_CONCURRENT_DOWNLOADS = 8    # simultaneous aiohttp streams
MAX_RETRIES = 3                 # attempts per file on 429/5xx/timeout
RETRY_BACKOFF_SECONDS = (1, 2, 4)
REQUEST_TIMEOUT_SECONDS = 60    # per-request read timeout

# --- Parquet -----------------------------------------------------------------------
PARQUET_COMPRESSION = "zstd"    # best ratio for tick data

# --- aggTrades schema (archive column order) ---------------------------------------
AGGTRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",   # milliseconds since epoch
    "is_buyer_maker",  # True -> seller aggressor (b_t = -1), False -> buyer (b_t = +1)
]
AGGTRADES_DTYPES = {
    "agg_trade_id": "int64",
    "price": "float64",
    "quantity": "float64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "transact_time": "int64",
    "is_buyer_maker": "bool",
}


# --- Path/URL helpers ---------------------------------------------------------------
def monthly_url(symbol: str, year_month: str) -> str:
    """Remote URL of a symbol's monthly aggTrades zip."""
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{year_month}.zip"


def zip_path(symbol: str, year_month: str) -> str:
    """Local path where a symbol's monthly zip is stored."""
    return f"{RAW_DIR}/{symbol}/{symbol}-aggTrades-{year_month}.zip"


def parquet_path(symbol: str, year_month: str) -> str:
    """Local path where a symbol's monthly parquet is stored."""
    return f"{PARQUET_DIR}/{symbol}/{symbol}-aggTrades-{year_month}.parquet"
