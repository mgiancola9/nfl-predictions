import os
import numpy as np
import pandas as pd
from glob import glob

# Paths (relative to src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Load & filter games
def load_games():
    """
    Load games.csv, keep REG season only, seasons 2021-2025.
    We'll train on 2021-2024 and evaluate on 2025.

    IMPORTANT:
      Assumes `spread_line` is from the AWAY perspective:
        spread_line < 0  -> away is favorite by |spread_line|
        spread_line > 0  -> home is favorite by spread_line
    """
    path = os.path.join(DATA_DIR, "games.csv")
    games = pd.read_csv(path)

    required_cols = [
        "season", "game_type", "home_team", "away_team",
        "home_score", "away_score", "spread_line"
    ]
    for c in required_cols:
        if c not in games.columns:
            raise ValueError(f"Missing '{c}' in games.csv. Found: {games.columns.tolist()}")

    # Regular season only
    games = games[games["game_type"] == "REG"].copy()

    # Filter modeling window
    games = games[games["season"].between(2021, 2025)].copy()

    print("Games by season (REG only):")
    print(games["season"].value_counts().sort_index())
    print(f"Games shape after filter: {games.shape}")

    return games


# 2. Load team stats
def load_team_stats():
    """
    Load team-level regular season stats for 2021-2024 from:
      stats_team_reg_2021.csv, ..., stats_team_reg_2024.csv
    """
    pattern = os.path.join(DATA_DIR, "stats_team_reg_*.csv")
    files = sorted(glob(pattern))

    dfs = []
    for f in files:
        year = int(os.path.basename(f).split("_")[-1].split(".")[0])

        if 2021 <= year <= 2024:
            df = pd.read_csv(f)

            if "season_type" in df.columns:
                df = df[df["season_type"] == "REG"].copy()

            if "season" not in df.columns:
                df["season"] = year

            dfs.append(df)

    if not dfs:
        raise ValueError("No team stats loaded. Check stats_team_reg_*.csv for 2021-2024.")

    team = pd.concat(dfs, ignore_index=True)

    if team.columns.duplicated().any():
        team = team.loc[:, ~team.columns.duplicated()]

    if "team" in team.columns:
        team = team.rename(columns={"team": "team_name"})
    elif "recent_team" in team.columns:
        team = team.rename(columns={"recent_team": "team_name"})
    else:
        raise ValueError(f"No team identifier column in team stats. Columns: {team.columns.tolist()}")

    print(f"Team stats combined shape: {team.shape}")
    return team


# 3. Load + aggregate player stats
def load_player_team_agg():
    """
    Load player-level stats for 2021-2024 (per-year files),
    restrict to REG if available, aggregate numerics to (season, team_name).
    """
    pattern = os.path.join(DATA_DIR, "stats_player_reg_*.csv")
    files = sorted(glob(pattern))

    dfs = []
    for f in files:
        year = int(os.path.basename(f).split("_")[-1].split(".")[0])

        if 2021 <= year <= 2024:
            df = pd.read_csv(f)

            if "season_type" in df.columns:
                df = df[df["season_type"] == "REG"].copy()

            if "season" not in df.columns:
                df["season"] = year

            dfs.append(df)

    if not dfs:
        print("No player stats files found for 2021-2024. Skipping player-based features.")
        return None

    players = pd.concat(dfs, ignore_index=True)

    if players.columns.duplicated().any():
        players = players.loc[:, ~players.columns.duplicated()]

    if "season" not in players.columns:
        raise ValueError(f"'season' missing in player stats after concat.")

    team_col = None
    for cand in ["recent_team", "team"]:
        if cand in players.columns:
            team_col = cand
            break

    if team_col is None:
        print("No team column in player stats; skipping player aggregation.")
        return None

    numeric_cols = players.select_dtypes(include="number").columns.tolist()
    group_cols = ["season", team_col]
    numeric_cols = [c for c in numeric_cols if c not in group_cols]

    agg = (
        players[group_cols + numeric_cols]
        .groupby(group_cols, as_index=False)
        .mean()
    )

    agg = agg.rename(columns={team_col: "team_name"})
    print(f"Player aggregated (team-level) shape: {agg.shape}")
    return agg

# 4. Build team feature table
def build_team_features(team_stats, player_agg):
    """
    Merge raw team stats and aggregated player stats.
    Key: (season, team_name)
    """
    if player_agg is None:
        print("No player aggregation; using only team stats.")
        return team_stats

    feats = pd.merge(
        team_stats,
        player_agg,
        on=["season", "team_name"],
        how="left",
        suffixes=("_team", "_player"),
    )

    print(f"Final team feature table shape: {feats.shape}")
    return feats

# 5. Merge team features into games
def merge_into_games(games, team_features):
    """
    Attach home/away team features to each game based on (season, team).
    Assumes:
      games.home_team / games.away_team match team_features.team_name.
    """
    for c in ["home_team", "away_team", "season"]:
        if c not in games.columns:
            raise ValueError(f"Missing '{c}' in games for merge.")

    # Home merge
    merged = games.merge(
        team_features,
        left_on=["season", "home_team"],
        right_on=["season", "team_name"],
        how="left",
        suffixes=("", "_home"),
    )

    for col in list(merged.columns):
        if (
            col not in games.columns
            and col != "team_name"
            and not col.endswith("_home")
        ):
            merged.rename(columns={col: f"{col}_home"}, inplace=True)

    # Away merge
    merged = merged.merge(
        team_features,
        left_on=["season", "away_team"],
        right_on=["season", "team_name"],
        how="left",
        suffixes=("", "_away"),
    )

    for col in list(merged.columns):
        if (
            col not in games.columns
            and not col.endswith("_home")
            and col != "team_name"
            and not col.endswith("_away")
        ):
            merged.rename(columns={col: f"{col}_away"}, inplace=True)

    drop_teamname_cols = [c for c in merged.columns if c.startswith("team_name")]
    merged = merged.drop(columns=drop_teamname_cols, errors="ignore")

    print(f"Merged games + features shape: {merged.shape}")
    return merged


# 6. Create labels & prune
def add_labels(df):
    """
    Define ATS target: fav_cover

    Conventions:
      - spread_line is from AWAY perspective:
          s < 0  -> away is favorite by |s|
          s > 0  -> home is favorite by s

    Let:
      home_margin = home_score - away_score   (or `result` if present)

    Favorite covers if:
      - away favorite (s < 0): home_margin < s        (away wins by > |s|)
      - home favorite (s > 0): home_margin > s        (home wins by > s)
      - s == 0: no favorite -> fav_cover = 0
    """
    if "result" in df.columns:
        home_margin = df["result"]
    else:
        if "home_score" not in df.columns or "away_score" not in df.columns:
            raise ValueError("Missing scores for label construction.")
        home_margin = df["home_score"] - df["away_score"]

    if "spread_line" not in df.columns:
        raise ValueError("Missing 'spread_line' for ATS labels.")

    s = df["spread_line"]

    fav_cover = np.where(
        s < 0,
        (home_margin < s).astype(int),              # away favorite covers
        np.where(
            s > 0,
            (home_margin > s).astype(int),          # home favorite covers
            0                                       # pick them -> no favorite
        )
    )

    df["home_margin"] = home_margin
    df["fav_cover"] = fav_cover

    return df


def final_feature_prune(df):
    """
    Drop IDs / obvious leak columns.
    Keep fav_cover as the supervised target.
    """
    drop_cols = [
        "game_id",
        "game_type",
        "gameday",
        "weekday",
        "gametime",
        "old_game_id",
        "gsis",
        "nfl_detail_id",
        "pfr",
        "pff",
        "espn",
        "ftn",
        "away_qb_id",
        "home_qb_id",
        "away_qb_name",
        "home_qb_name",
        "away_coach",
        "home_coach",
        "referee",
        "stadium_id",
        "stadium",
        "home_score",
        "away_score",
        # keep home_margin (already used to build label) only if wanted for analysis;
        # but we drop it here to avoid leakage into features:
        "home_margin",
    ]

    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


# 7. Main pipeline
def main():
    games = load_games()
    team_stats = load_team_stats()
    player_agg = load_player_team_agg()
    team_features = build_team_features(team_stats, player_agg)

    # Time-based split: 2021-2024 train, 2025 test
    train_games = games[games["season"] < 2025].copy()
    test_games = games[games["season"] == 2025].copy()

    train_merged = merge_into_games(train_games, team_features)
    test_merged = merge_into_games(test_games, team_features)

    train_labeled = add_labels(train_merged)
    test_labeled = add_labels(test_merged)

    train_final = final_feature_prune(train_labeled)
    test_final = final_feature_prune(test_labeled)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_processed.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_processed.csv")

    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)

    print(" Preprocessing complete.")
    print(f"Train: {train_final.shape} → {train_path}")
    print(f"Test:  {test_final.shape} → {test_path}")


if __name__ == "__main__":
    main()
