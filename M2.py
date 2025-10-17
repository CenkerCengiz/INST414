import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List
from matplotlib.patches import Patch

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR       = Path(r"C:\Users\cenke\M2")     
EXTRA_CSV_DIR  = DATA_DIR / "extra_csv"         

GOLD_FILE      = DATA_DIR / "gold_price.csv"
SILVER_FILE    = DATA_DIR / "LBMA-SILVER.csv"
INDICES_FILE   = DATA_DIR / "INDICES_DATA.csv"

START_DATE     = "2005-01-01"   
MIN_OBS        = 120            
EDGE_THRESHOLD = 0.50           
DENSE_THRESHOLD= 0.20                

def make_unique(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
    return out

def outer_merge_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["Date"])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="Date", how="outer")
    return merged

gold = pd.read_csv(GOLD_FILE, parse_dates=["date"])
gold = gold.rename(columns={"date": "Date", "price": "Gold"}).dropna(subset=["Gold"]).sort_values("Date")

silver_raw = pd.read_csv(SILVER_FILE, parse_dates=["Date"])
silver = silver_raw[["Date", "USD"]].rename(columns={"USD": "Silver"}).dropna().sort_values("Date")

indices_raw = pd.read_csv(INDICES_FILE, low_memory=False)
subheaders = indices_raw.iloc[0].tolist()
orig_cols  = indices_raw.columns.tolist()

new_cols, current_ticker = [], None
for col_name, sub in zip(orig_cols, subheaders):
    if not str(col_name).startswith("Unnamed"):
        current_ticker = str(col_name).strip()
    sub_clean = str(sub).strip().replace(" ", "_")
    new_cols.append("Date" if sub_clean.lower() == "date" else f"{current_ticker}_{sub_clean}")

indices_df = indices_raw[1:].copy()
indices_df.columns = new_cols
indices_df["Date"] = pd.to_datetime(indices_df["Date"], errors="coerce")
indices_df = indices_df.dropna(subset=["Date"]).sort_values("Date")

adj_close_cols = [c for c in indices_df.columns if c.endswith("_Adj_Close")]
close_cols     = [c for c in indices_df.columns if c.endswith("_Close") and not c.endswith("_Adj_Close")]

selected_price_cols: Dict[str, str] = {}
for c in adj_close_cols:
    selected_price_cols[c.replace("_Adj_Close", "")] = c
for c in close_cols:
    t = c.replace("_Close", "")
    if t not in selected_price_cols:
        selected_price_cols[t] = c

# Keep duplicates with suffixes so nothing is lost
rename_map = {}
used_names = []
for ticker, col in selected_price_cols.items():
    used_names.append(ticker)
unique_names = make_unique(used_names)

for (ticker, col), uname in zip(selected_price_cols.items(), unique_names):
    rename_map[col] = uname

indices_prices = indices_df[["Date"] + list(selected_price_cols.values())].copy()
indices_prices = indices_prices.rename(columns=rename_map)
for t in [c for c in indices_prices.columns if c != "Date"]:
    indices_prices[t] = pd.to_numeric(indices_prices[t], errors="coerce")

extra_frames = []
if EXTRA_CSV_DIR.exists():
    for csv_path in EXTRA_CSV_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            price_col = None
            for candidate in ["Adj Close", "AdjClose", "Adj_Close", "Close", "close", "adj_close"]:
                if candidate in df.columns:
                    price_col = candidate
                    break
            if price_col is None:
                print(f"[skip] {csv_path.name}: no Adj Close/Close column found")
                continue
            sym = csv_path.stem  
            df2 = df[["Date", price_col]].rename(columns={price_col: sym})
            extra_frames.append(df2)
            print(f"[extra_csv] added {sym} from {csv_path.name}")
        except Exception as e:
            print(f"[skip] {csv_path.name}: {e}")
else:
    print(f"[info] extra CSV dir not found: {EXTRA_CSV_DIR} (create it to add more assets)")

extra_data_offline = outer_merge_frames(extra_frames) if extra_frames else pd.DataFrame(columns=["Date"])

extra_data_online = pd.DataFrame(columns=["Date"])
try:
    import yfinance as yf
    # sector ETFs + currencies + crypto
    sector_etfs       = ["XLE","XLB","XLF","XLK","XLY","XLP","XLV","XLI","XLU","XLRE"]
    currencies_crypto = ["DX=F", "EURUSD=X", "BTC-USD", "ETH-USD"]  
    extra_tickers     = sector_etfs + currencies_crypto

    raw = yf.download(
        extra_tickers,
        start=START_DATE,
        auto_adjust=False,        # ensure 'Adj Close' exists
        group_by="column",
        progress=False,
        threads=True,
    )

    # Extract a wide table of prices
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            adj = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            adj = raw["Close"].copy()
        else:
            first_level = raw.columns.levels[0][0]
            adj = raw.xs(first_level, axis=1, level=0, drop_level=True)
    else:
        adj = raw.copy()

    adj = adj.dropna(how="all", axis=1).reset_index()
    adj.rename(columns={"Date": "Date"}, inplace=True)
    # Make column names unique if duplicates appear
    adj.columns = make_unique(list(adj.columns))
    extra_data_online = adj
    print(f"[yfinance] merged {extra_data_online.shape[1]-1} extra series (out of {len(extra_tickers)})")
except Exception as e:
    print("[yfinance] skip (not installed / fetch failed):", e)

# Choose which extra data to use: combine offline and online if both exist
extra_data_list = []
if not extra_data_offline.empty:
    extra_data_list.append(extra_data_offline)
if not extra_data_online.empty:
    extra_data_list.append(extra_data_online)
extra_data = outer_merge_frames(extra_data_list) if extra_data_list else pd.DataFrame(columns=["Date"])

merged = (
    gold[["Date", "Gold"]]
    .merge(silver[["Date", "Silver"]], on="Date", how="outer")
    .merge(indices_prices, on="Date", how="outer")
    .merge(extra_data, on="Date", how="outer")
    .sort_values("Date")
)

# Fill first, then drop truly sparse
merged = merged[merged["Date"] >= pd.Timestamp(START_DATE)].set_index("Date").sort_index().ffill().bfill().reset_index()
valid_cols = ["Date"] + [c for c in merged.columns if c != "Date" and merged[c].count() >= MIN_OBS]
merged = merged[valid_cols]

rets = merged.set_index("Date").pct_change().dropna(how="all")
stds = rets.std()
keep_assets = stds[stds > 0].index.tolist()
rets = rets[keep_assets]
corr = rets.corr()

# Focused correlation with Gold and Silver 
gold_corr = corr["Gold"].sort_values(ascending=False).drop("Gold")
silver_corr = corr["Silver"].sort_values(ascending=False).drop("Silver")

print("\n=== Correlation with GOLD ===")
print(gold_corr.head(10))
print("\n=== Correlation with SILVER ===")
print(silver_corr.head(10))

# Show assets that move opposite to gold/silver
print("\n=== Negative correlation with GOLD ===")
print(gold_corr.tail(10))
print("\n=== Negative correlation with SILVER ===")
print(silver_corr.tail(10))

print("\n[diag] counts at each stage")
print("  indices_prices cols (excluding Date):", len(indices_prices.columns) - 1)
print("  extra_csv cols (excluding Date):", 0 if extra_data_offline.empty else len(extra_data_offline.columns) - 1)
print("  yfinance cols (excluding Date):", 0 if extra_data_online.empty else len(extra_data_online.columns) - 1)
print("  merged cols (excluding Date):", len(merged.columns) - 1)
print("  final nodes (corr columns):", len(rets.columns))

G = nx.Graph()
for i, a in enumerate(corr.columns):
    G.add_node(a)
    for j in range(i + 1, len(corr.columns)):
        b = corr.columns[j]
        w = float(corr.iloc[i, j])
        if np.isfinite(w) and abs(w) >= EDGE_THRESHOLD:
            G.add_edge(a, b, weight=w)

deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)
try:
    eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
except nx.NetworkXException:
    eig_cent = {n: np.nan for n in G.nodes}

centrality_df = pd.DataFrame({
    "degree_centrality": pd.Series(deg_cent),
    "betweenness_centrality": pd.Series(bet_cent),
    "eigenvector_centrality": pd.Series(eig_cent),
}).sort_values(["eigenvector_centrality", "degree_centrality"], ascending=False)

print("\n=== Top 15 nodes by eigenvector centrality (base network) ===")
print(centrality_df.head(15))

# Simple network plot 
pos = nx.spring_layout(G, seed=42, k=0.6, iterations=200)
plt.figure(figsize=(16, 14), dpi=250)
nx.draw(
    G, pos,
    node_size=400,
    with_labels=True,
    font_size=7,
    width=[1 + 2.5 * abs(G[u][v]['weight']) for u, v in G.edges()],
    alpha=0.95
)
plt.title(f"Asset Network (|corr| ≥ {EDGE_THRESHOLD}) — clean view")
plt.axis("off")
plt.savefig(DATA_DIR / "asset_network_simple.png", dpi=300, bbox_inches="tight")
plt.close()
print("[saved] asset_network_simple.png")

G_full = nx.Graph()
for i, a in enumerate(corr.columns):
    G_full.add_node(a)
    for j in range(i + 1, len(corr.columns)):
        b = corr.columns[j]
        w = float(corr.iloc[i, j])
        if np.isfinite(w) and abs(w) >= DENSE_THRESHOLD:
            G_full.add_edge(a, b, weight=w)

print(f"\nDense graph summary: nodes={G_full.number_of_nodes()}, edges={G_full.number_of_edges()} (threshold={DENSE_THRESHOLD})")

# Centralities on dense graph (for sizing/labels)
try:
    eig_full = nx.eigenvector_centrality(G_full, max_iter=1000)
except nx.NetworkXException:
    eig_full = {n: 0.0 for n in G_full.nodes}
deg_full = nx.degree_centrality(G_full)

# Categorize nodes for colors
def categorize(n: str) -> str:
    if n in ("Gold", "Silver"): return "Commodity"
    if n in ["BTC-USD", "ETH-USD"]: return "Crypto"
    if n in ["DX=F", "EURUSD=X", "DX-Y.NYB"]: return "Currency"
    if n in ["XLE","XLB","XLF","XLK","XLY","XLP","XLV","XLI","XLU","XLRE"]: return "Sector ETF"
    if n.startswith("^") or n.endswith(".SS") or n.endswith(".SZ") or n.endswith(".TA") or n.endswith(".JO"): return "Index"
    return "Other"

palette = {
    "Commodity": "#FF7F0E",   
    "Sector ETF": "#1F77B4",  
    "Currency":  "#2CA02C",   
    "Crypto":    "#9467BD",   
    "Index":     "#D62728",   
    "Other":     "#8C564B"    
}
cats = {n: categorize(n) for n in G_full.nodes}
node_colors = [palette[cats[n]] for n in G_full.nodes]

# Node sizes + layout tweaks for visibility
eig_vals = np.array([eig_full.get(n, 0.0) for n in G_full.nodes])
sizes = 150 + 1200 * (eig_vals / eig_vals.max()) if eig_vals.max() > 0 else 150
pos_full = nx.spring_layout(G_full, seed=42, k=0.6, iterations=200)

# Draw edges/nodes
plt.figure(figsize=(30, 26), dpi=300)
nx.draw_networkx_edges(
    G_full, pos_full,
    width=[0.8 + 2.5 * abs(G_full[u][v]['weight']) for u, v in G_full.edges()],
    alpha=0.25
)
nx.draw_networkx_nodes(
    G_full, pos_full,
    node_color=node_colors,
    node_size=sizes,
    linewidths=0.6,
    edgecolors="#222",
    alpha=0.95
)

highlight_edges = [(u, v) for u, v in G_full.edges() if u in ["Gold","Silver"] or v in ["Gold","Silver"]]
nx.draw_networkx_edges(
    G_full, pos_full,
    edgelist=highlight_edges,
    width=2.5,
    edge_color=["#FFD700" if "Gold" in e else "#C0C0C0" for e in highlight_edges],
    alpha=0.8
)

highlight_edges = [(u, v) for u, v in G_full.edges() if u in ["Gold","Silver"] or v in ["Gold","Silver"]]
nx.draw_networkx_edges(
    G_full, pos_full,
    edgelist=highlight_edges,
    width=2.5,
    edge_color=["#FFD700" if "Gold" in e else "#C0C0C0" for e in highlight_edges],
    alpha=0.8
)
print("Top 10 most connected assets:")
print(centrality_df.head(10))
targets = [t for t in ["XLE","XLB","XLF","XLK","XLY","XLP","XLV","XLI","XLU","XLRE"] if t in corr.columns]
print("\nSectors vs GOLD (desc):\n", corr.loc[targets, "Gold"].sort_values(ascending=False))
print("\nSectors vs SILVER (desc):\n", corr.loc[targets, "Silver"].sort_values(ascending=False))
prices = merged.set_index("Date")
roll_gold_eq = prices["Gold"].pct_change().rolling(90).corr(prices["^GSPC"].pct_change())
roll_gold_usd = prices["Gold"].pct_change().rolling(90).corr(prices["DX=F"].pct_change())
roll_silver_usd = prices["Silver"].pct_change().rolling(90).corr(prices["DX=F"].pct_change())

import math

def df_to_png(df: pd.DataFrame, title: str, save_path: Path, col_width=2.4, row_height=0.45, 
              fontsize=10, title_size=14, header_color="#ECECEC", edge_color="#CCCCCC"):

    fmt_df = df.copy()
    for c in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[c]):
            fmt_df[c] = fmt_df[c].map(lambda x: f"{x:,.3f}")

    n_rows, n_cols = fmt_df.shape

    fig_w = max(6, n_cols * col_width)
    fig_h = max(2.5, (n_rows + 1) * row_height + 1.0)  

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")

    ax.text(0, 1.05, title, fontsize=title_size, fontweight="bold", transform=ax.transAxes)

    cell_text = [list(fmt_df.columns)] + fmt_df.values.tolist()

    the_table = plt.table(
        cellText=cell_text,
        cellLoc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]  
    )

    for j in range(n_cols):
        cell = the_table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_edgecolor(edge_color)
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(fontsize)

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = the_table[(i, j)]
            cell.set_edgecolor(edge_color)
            cell.get_text().set_fontsize(fontsize)

    for j in range(n_cols):
        the_table.auto_set_column_width(j)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")

# Top 10 central assets (use your existing centrality_df)
top10 = (centrality_df
         .reset_index()
         .rename(columns={"index": "Asset",
                          "degree_centrality": "Degree",
                          "betweenness_centrality": "Betweenness",
                          "eigenvector_centrality": "Eigenvector"})
         .loc[:, ["Asset", "Eigenvector", "Degree", "Betweenness"]]
         .head(10))

df_to_png(
    top10,
    title="Top 10 Central Assets (Eigenvector, Degree, Betweenness)",
    save_path=DATA_DIR / "table_top10_central_assets.png",
    col_width=2.6
)

# Key correlations table (Gold/Silver focus)
pairs = []
def safe_corr(a, b):
    return corr.loc[a, b] if (a in corr.index and b in corr.columns) else np.nan

pairs.append(("Gold – Silver", safe_corr("Gold", "Silver")))
pairs.append(("Gold – USD Index (DX=F)", safe_corr("Gold", "DX=F")))
pairs.append(("Silver – USD Index (DX=F)", safe_corr("Silver", "DX=F")))
pairs.append(("Gold – S&P 500 (^GSPC)", safe_corr("Gold", "^GSPC")))
pairs.append(("Silver – S&P 500 (^GSPC)", safe_corr("Silver", "^GSPC")))
pairs.append(("Gold – BTC-USD", safe_corr("Gold", "BTC-USD")))
pairs.append(("Silver – XLB (Materials)", safe_corr("Silver", "XLB")))

key_corr_df = pd.DataFrame(pairs, columns=["Asset Pair", "Correlation"])

df_to_png(
    key_corr_df,
    title="Key Correlations (Daily Returns)",
    save_path=DATA_DIR / "table_key_correlations.png",
    col_width=3.2
)

# Sector correlations vs Gold/Silver (only those that exist in corr)
sector_list = [x for x in ["XLE","XLB","XLF","XLK","XLY","XLP","XLV","XLI","XLU","XLRE"] if x in corr.index]
sector_df = pd.DataFrame({
    "Sector ETF": sector_list,
    "Corr with Gold": [safe_corr(s, "Gold") for s in sector_list],
    "Corr with Silver": [safe_corr(s, "Silver") for s in sector_list],
}).sort_values("Corr with Gold", ascending=False).reset_index(drop=True)

df_to_png(
    sector_df,
    title="Sector ETFs vs Gold and Silver (Correlation)",
    save_path=DATA_DIR / "table_sector_corrs_gold_silver.png",
    col_width=2.6
)

plt.figure(figsize=(12,5), dpi=150)
plt.plot(roll_gold_eq.index, roll_gold_eq.values, label="Gold vs S&P 500")
plt.plot(roll_gold_usd.index, roll_gold_usd.values, label="Gold vs USD Index (DX=F)")
plt.plot(roll_silver_usd.index, roll_silver_usd.values, label="Silver vs USD Index (DX=F)")
plt.axhline(0, color='k', lw=0.5)
plt.legend(); plt.title("Rolling 90-day correlations"); plt.tight_layout()
plt.savefig(DATA_DIR / "rolling_corr_focus.png", dpi=200)
plt.close()
legend_elems = [Patch(facecolor=v, edgecolor="#222", label=k) for k, v in palette.items()]
plt.legend(handles=legend_elems, loc="upper left")
plt.title(f"Full Asset Network (|corr| ≥ {DENSE_THRESHOLD}) — sectors, currencies, crypto, commodities")
plt.axis("off")
plt.savefig(DATA_DIR / "asset_network_full_hires.png", dpi=300, bbox_inches="tight")
plt.close()
print("[saved] asset_network_full_hires.png")