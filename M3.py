import pandas as pd
import numpy as np
import yfinance as yf
import re
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

plt.rcParams["figure.dpi"] = 130

LOCAL_PATH = r"C:\Users\cenke\Downloads\symbols_valid_meta.csv"   
QUERIES = ["AAPL", "XOM", "JPM"]        
MIN_OBS = 150                          
YEARS_BACK = 5                         

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

def load_sp500_from_local(path=LOCAL_PATH) -> pd.DataFrame:
    """
    Load local CSV with at least a 'ticker' column, ideally 'name' and 'sector'.
    Normalizes column names and tickers to yfinance format.
    """
    df = pd.read_csv(path)

    rename_map = {
        "Symbol": "ticker", "Ticker": "ticker", "ticker": "ticker",
        "Security": "name", "Company": "name", "Name": "name", "name": "name",
        "GICS Sector": "sector", "Sector": "sector", "sector": "sector"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["ticker", "name", "sector"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["ticker"] = (
        df["ticker"].astype(str)
        .str.replace(".", "-", regex=False)
        .str.upper()
        .str.strip()
    )

    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df[["ticker", "name", "sector"]]

def backfill_sectors_with_yf(sp500: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    """
    Fill missing sector values via yfinance .info (parallelized).
    """
    missing = sp500.loc[sp500["sector"].isna(), "ticker"].tolist()
    if not missing:
        return sp500

    print(f"[INFO] Backfilling sector for {len(missing)} tickers via yfinance…")
    import concurrent.futures as cf

    def get_sector(t):
        try:
            info = yf.Ticker(t).info or {}
            return t, info.get("sector")
        except Exception:
            return t, None

    results = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for t, sec in ex.map(get_sector, missing):
            results.append((t, sec))

    fill_df = pd.DataFrame(results, columns=["ticker", "sector_fill"])
    sp500 = sp500.merge(fill_df, on="ticker", how="left")
    sp500["sector"] = sp500["sector"].fillna(sp500["sector_fill"])
    sp500 = sp500.drop(columns=["sector_fill"])
    print(f"Sectors filled: {sp500['sector'].notna().sum()} of {len(sp500)}")
    return sp500

def extract_adj_close(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly extract a wide Adj Close table from yfinance output.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        tops = raw.columns.get_level_values(0)
        if "Adj Close" in tops:
            return raw["Adj Close"].copy()
        elif "Close" in tops:
            return raw["Close"].copy()
        else:
            first_level = raw.columns.levels[0][0]
            return raw.xs(first_level, axis=1, level=0, drop_level=True)
    return raw.copy()

def top_similar(returns_df: pd.DataFrame, query: str, k: int = 10, meta_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Top-k by Pearson correlation with query; excludes self; merges name/sector if meta provided.
    """
    if query not in returns_df:
        raise KeyError(f"Query {query} not found in returns data.")
    sims = returns_df.corrwith(returns_df[query]).dropna().sort_values(ascending=False)
    sims = sims[sims.index != query]
    out = sims.head(k).to_frame("correlation").reset_index().rename(columns={"index": "ticker"})
    if meta_df is not None:
        out = out.merge(meta_df[["ticker", "name", "sector"]], on="ticker", how="left")
    return out

sp500_ref = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
).rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
sp500_ref["ticker"] = sp500_ref["ticker"].str.replace(".", "-", regex=False).str.strip()

sp500_local = load_sp500_from_local(LOCAL_PATH)
sp500 = sp500_ref[sp500_ref["ticker"].isin(sp500_local["ticker"])].copy()
if len(sp500) < 450:
    print(f"[INFO] Intersection is small ({len(sp500)}). Using full S&P 500 reference list instead.")
    sp500 = sp500_ref.copy()

if "ticker" not in sp500.columns:
    sp500 = sp500.reset_index().rename(columns={"index": "ticker"})
for col in ["name", "sector"]:
    if col not in sp500.columns:
        sp500[col] = pd.NA
sp500 = sp500[["ticker", "name", "sector"]]

print(f"Using S&P 500 reference list. Tickers: {len(sp500)}; sectors known: {sp500['sector'].notna().sum()}")

import os, time
from pathlib import Path

END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - DateOffset(years=YEARS_BACK)

CACHE_PATH = Path("prices_5y.csv")

def load_cache():
    if CACHE_PATH.exists():
        try:
            df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            print(f"[CACHE] Loaded cached prices: {df.shape}")
            return df
        except Exception as e:
            print(f"[CACHE] Failed to read cache: {e}")
    return None

def save_cache(df):
    try:
        df.to_csv(CACHE_PATH)
        print(f"[CACHE] Saved prices cache: {df.shape}")
    except Exception as e:
        print(f"[CACHE] Failed to write cache: {e}")

def polite_download(tickers, start, end, max_retries=4, pause=4.0):
    """Small batch download with retries (threads off to avoid rate limits)."""
    attempt = 0
    while True:
        try:
            df = yf.download(
                tickers,
                start=str(start.date()),
                end=str(end.date()),
                auto_adjust=False,
                group_by="column",
                progress=False,
                threads=False,      
                interval="1d",
                timeout=30,
            )
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                tops = df.columns.get_level_values(0)
                if "Adj Close" in tops:
                    out = df["Adj Close"].copy()
                elif "Close" in tops:
                    out = df["Close"].copy()
                else:
                    out = df.xs(df.columns.levels[0][0], axis=1, level=0, drop_level=True)
            else:
                out = df.copy()
            return out
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                print(f"[WARN] Batch {tickers[:3]}… failed after {attempt} attempts: {e}")
                return pd.DataFrame()
            sleep_s = pause * (2 ** (attempt - 1))
            print(f"[INFO] Retry {attempt}/{max_retries} in {sleep_s:.1f}s — reason: {e}")
            time.sleep(sleep_s)

def download_in_chunks(all_tickers, start, end, chunk_size=15):
    """Download all tickers in chunks; prioritize queries first."""
    qs = [t for t in QUERIES if t in all_tickers]
    rest = [t for t in all_tickers if t not in qs]
    ordered = qs + rest

    combined = pd.DataFrame()
    for i in range(0, len(ordered), chunk_size):
        batch = ordered[i:i+chunk_size]
        print(f"[DL] {i+1}-{i+len(batch)} / {len(ordered)} …")
        part = polite_download(batch, start, end)
        if not part.empty:
            combined = part if combined.empty else combined.join(part, how="outer")
        time.sleep(1.5)  
    return combined

universe = sorted(set(sp500["ticker"]))
cached = load_cache()

if cached is not None:
    have = [c for c in cached.columns if c in universe]
    need = [t for t in universe if t not in have]
    adj = cached[have].copy()
    if need:
        print(f"[CACHE] Need {len(need)} more tickers; downloading missing…")
        more = download_in_chunks(need, START_DATE, END_DATE, chunk_size=15)
        if not more.empty:
            adj = adj.join(more, how="outer")
            save_cache(adj)
else:
    print(f"Downloading prices for {len(universe)} tickers from {START_DATE.date()} to {END_DATE.date()}…")
    adj = download_in_chunks(universe, START_DATE, END_DATE, chunk_size=15)
    if not adj.empty:
        save_cache(adj)

adj = adj.loc[:, adj.count() >= MIN_OBS].dropna(how="all")
print(f"Adj Close shape after cleaning: {adj.shape}")

missing_q = [q for q in QUERIES if q not in adj.columns]
if missing_q:
    print(f"[WARN] Queries missing {missing_q} — retrying those explicitly…")
    qdf = download_in_chunks(missing_q, START_DATE, END_DATE, chunk_size=len(missing_q))
    if not qdf.empty:
        adj = adj.join(qdf, how="outer")
        adj = adj.loc[:, adj.count() >= MIN_OBS].dropna(how="all")
        save_cache(adj)

present_q = [q for q in QUERIES if q in adj.columns]
if len(present_q) < 1:
    raise ValueError("Failed to fetch any query tickers after retries. Re-run in a few minutes (rate limit).")
elif len(present_q) < len(QUERIES):
    print(f"[INFO] Proceeding with queries present: {present_q} (some still missing).")

returns = (
    adj.sort_index()
       .pct_change(fill_method=None)  
       .replace([np.inf, -np.inf], np.nan)
       .dropna(how="all")
)

queries = [q for q in QUERIES if q in returns.columns]
if not queries:
    raise ValueError("None of the query tickers are present in the downloaded data.")
returns_q = returns.dropna(subset=queries, how="any")
print(f"Returns shape after aligning on queries: {returns_q.shape}")

all_results = {}
for q in queries:
    topk = top_similar(returns_q, q, k=10, meta_df=sp500)
    all_results[q] = topk

    csv_path = f"top10_similar_{q}.csv"
    topk.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    plt.figure(figsize=(9, 4.5))
    plt.bar(topk["ticker"], topk["correlation"])
    plt.title(f"Top 10 Most Similar to {q} ({YEARS_BACK}Y daily-return correlation)")
    plt.ylabel("Correlation (Pearson)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = f"top10_similar_{q}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"[Saved] {fig_path}")

combined = (
    pd.concat([df.assign(query=q) for q, df in all_results.items()], ignore_index=True)
      .loc[:, ["query","ticker","correlation","name","sector"]]
)
combined.to_csv("top10_similar_all_queries.csv", index=False)
print("[Saved] top10_similar_all_queries.csv")

present = pd.Index(returns_q.columns)
meta = sp500[sp500["ticker"].isin(present)].copy()

ticker_to_sector = meta.set_index("ticker")["sector"]
sector_returns = (
    returns_q.T.join(ticker_to_sector.rename("sector"))
              .groupby("sector")
              .mean()
              .T
)
sector_returns = sector_returns.dropna(axis=1, how="all")
print(f"Sector returns shape: {sector_returns.shape}")

sector_corr = sector_returns.corr()
sector_corr.to_csv("sector_correlation_matrix.csv")
print("[Saved] sector_correlation_matrix.csv")

m = sector_corr.values.copy()
np.fill_diagonal(m, np.nan)
avg_corr = pd.Series(np.nanmean(m, axis=1), index=sector_corr.index).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(avg_corr.index, avg_corr.values)
plt.title(f"Average Correlation of Each Sector with All Other Sectors ({YEARS_BACK}Y Daily Returns)")
plt.xlabel("Mean Pearson correlation (excluding self)")
plt.tight_layout()
plt.savefig("sector_average_correlation_barplot.png", dpi=180)
plt.close()
print("[Saved] sector_average_correlation_barplot.png")

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    r = np.clip(sector_corr.values, -1, 1)
    dist = np.sqrt(2 * (1 - r))

    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")

    plt.figure(figsize=(8.5, 5.0))
    dendrogram(link, labels=sector_corr.columns.tolist(), leaf_rotation=45)
    plt.title(f"Sector Similarity Dendrogram ({YEARS_BACK}Y Returns)")
    plt.tight_layout()
    plt.savefig("sector_dendrogram.png", dpi=180)
    plt.close()
    print("[Saved] sector_dendrogram.png")
except Exception as e:
    print(f"[INFO] Skipping dendrogram (SciPy not available or error: {e})")
