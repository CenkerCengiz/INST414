import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fixtures = pd.read_csv(r"C:\Users\cenke\fixtures.csv")
teamStats = pd.read_csv(r"C:\Users\cenke\teamStats.csv")
leagues = pd.read_csv(r"C:\Users\cenke\leagues.csv")

top5 = ["ENG.1", "FRA.1", "ESP.1", "ITA.1", "GER.1"]
top5_leagues = leagues[leagues["midsizeName"].isin(top5)]["leagueId"].unique()
fixtures_top5 = fixtures[fixtures["leagueId"].isin(top5_leagues)]

df = pd.merge(
    teamStats,
    fixtures_top5[["eventId", "homeTeamId", "awayTeamId", "homeTeamWinner", "awayTeamWinner"]],
    on="eventId",
    how="inner"
)

def outcome(row):
    if row["teamId"] == row["homeTeamId"]:
        return 1 if row["homeTeamWinner"] else (0 if row["awayTeamWinner"] else None)
    elif row["teamId"] == row["awayTeamId"]:
        return 1 if row["awayTeamWinner"] else (0 if row["homeTeamWinner"] else None)
    else:
        return None

df["isWin"] = df.apply(outcome, axis=1)

df = df.dropna(subset=["isWin"])
df["isWin"] = df["isWin"].astype(int)

exclude_cols = ["seasonType", "eventId", "teamId", "teamOrder", "updateTime"]
stats_cols = [c for c in teamStats.columns if c not in exclude_cols]

for col in stats_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

subset = df[["isWin"] + stats_cols].copy()

correlations = subset.corr(method="pearson")["isWin"].drop("isWin").sort_values(ascending=False)
correlations_percent = correlations * 100

plt.figure(figsize=(12, 10))
sns.barplot(
    x=correlations_percent.values,
    y=correlations_percent.index,
    hue=correlations_percent.index,  
    palette="coolwarm",
    legend=False
)
plt.title("Correlation of Match Statistics with Winning (Top 5 Leagues)", fontsize=16)
plt.xlabel("Correlation with Winning (%)", fontsize=14)
plt.ylabel("Match Statistic", fontsize=14)
plt.tight_layout()

plt.savefig("correlation_bargraph_sorted.png", dpi=300, bbox_inches="tight")
plt.show()

print("Bar graph saved as correlation_bargraph_sorted.png")
