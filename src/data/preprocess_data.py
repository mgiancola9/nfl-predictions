import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# --- Directory Setup (Standard Data Science Structure) ---
# Assuming this script is run from project_root/src/data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

# Create output directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------
# 1. Loading Functions (I/O)
# ------------------------------------------------------------------------

def load_games() -> pd.DataFrame:
    """Load and filter the base games data."""
    path = DATA_DIR / "raw" / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing games.csv at: {path}")

    games = pd.read_csv(path)

    # Filter to Regular Season and desired modeling window (2021-2025)
    games = games[
        games["game_type"].eq("REG") &
        games["season"].between(2021, 2025)
    ].copy()

    # Cast team names to string to avoid merge issues
    games["home_team"] = games["home_team"].astype(str)
    games["away_team"] = games["away_team"].astype(str)

    # Clean up games with no spread line (e.g., historical or incomplete 2025 data)
    games = games.dropna(subset=['spread_line', 'home_score', 'away_score']).reset_index(drop=True)

    return games

def load_team_features() -> pd.DataFrame:
    """
    Load team features computed by team_strengths.py and build_features.py.
    This file should contain one row per (season, team) or (game_id).
    """
    # Placeholder: In a complete pipeline, you would run team_strengths.py 
    # and build_features.py first, which would save an interim file here.
    path = INTERIM_DIR / "team_strengths_features.csv"
    if not path.exists():
        print(f"WARNING: Team Strength features not found at {path}. Proceeding without them.")
        return pd.DataFrame()
    
    return pd.read_csv(path)

def load_market_features() -> pd.DataFrame:
    """
    Load market features computed by market_features.py (e.g., Line Movement, Implied Probabilities).
    """
    # Placeholder for NEW file
    path = INTERIM_DIR / "market_features.csv"
    if not path.exists():
        print(f"WARNING: Market features not found at {path}. Proceeding without them.")
        # NOTE: You will need to implement this to hit the >= 57% goal!
        return pd.DataFrame() 

    return pd.read_csv(path)

def load_injury_features() -> pd.DataFrame:
    """
    Load injury features computed by injury_features.py (e.g., Total VAM lost).
    """
    # Placeholder for NEW file
    path = INTERIM_DIR / "injury_features.csv"
    if not path.exists():
        print(f"WARNING: Injury features not found at {path}. Proceeding without them.")
        # NOTE: You will need to implement this to hit the >= 57% goal!
        return pd.DataFrame() 
        
    return pd.read_csv(path)


# ------------------------------------------------------------------------
# 2. Target Label Creation
# ------------------------------------------------------------------------

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the target variable 'fav_cover' based on the confirmed spread line perspective.
    spread_line < 0 means Away is favorite.
    spread_line > 0 means Home is favorite.
    """
    df = df.copy()

    # 1. Calculate the Home Team's Cover Margin (Against The Spread)
    # The margin is positive if the home team covered or beat the spread implied by 'spread_line'
    # Home ATS Margin = (Home Score - Away Score) + spread_line (from Away perspective)
    df["ats_margin_home"] = (df["home_score"] - df["away_score"]) + df["spread_line"]
    
    # 2. Determine if the Favorite Covered
    # Logic:
    # - If Home is Fav (spread_line > 0) AND Home Covered (ats_margin_home > 0) -> Favorite Covered (1)
    # - If Away is Fav (spread_line < 0) AND Away Covered (ats_margin_home < 0) -> Favorite Covered (1)
    # - If a team was favored but did NOT cover, or it was a Push (ats_margin_home == 0) -> Did Not Cover (0)

    # Identify the favorite:
    df["home_is_fav"] = df["spread_line"] > 0
    df["away_is_fav"] = df["spread_line"] < 0
    
    # Determine the outcome relative to the spread (ignoring push for now)
    df["home_cover"] = df["ats_margin_home"] > 0
    df["away_cover"] = df["ats_margin_home"] < 0
    
    # Target Label: fav_cover = 1 if the favored team won ATS
    df["fav_cover"] = np.where(
        (df["home_is_fav"] & df["home_cover"]) | 
        (df["away_is_fav"] & df["away_cover"]),
        1,  # Favorite covered
        0   # Favorite did not cover (or it was a push)
    )

    # 3. Handle Pushes (ats_margin_home == 0)
    # For classification, we usually assign pushes to the non-cover class (0) or drop them. 
    # Since we used np.where, pushes (where margin=0) defaulted to 0. We'll leave them as 0 for simplicity.

    return df.drop(columns=["home_is_fav", "away_is_fav", "home_cover", "away_cover"])


# ------------------------------------------------------------------------
# 3. Main Pipeline: Merge, Split, and Save
# ------------------------------------------------------------------------

def main():
    """Run the complete preprocessing pipeline."""
    print("--- 1. Loading Core Data ---")
    games = load_games()
    
    print("--- 2. Loading Feature Modules (Interim) ---")
    team_features = load_team_features()
    market_features = load_market_features()
    injury_features = load_injury_features()
    
    # List of all feature dataframes to merge
    all_features = [team_features, market_features, injury_features]
    
    # 3. Merging all features into the games DataFrame
    print("--- 3. Merging Features ---")
    merged_df = games.copy()
    
    for feature_df in all_features:
        if not feature_df.empty:
            # Assume features are keyed by 'game_id' or a combination of 'season', 'week', 'home_team', 'away_team'
            # Adjust the 'on' parameter based on how you key your features
            merged_df = merged_df.merge(
                feature_df, 
                on=["season", "week", "home_team", "away_team"], # Example keys
                how="left",
                suffixes=('', '_feat')
            )

    # 4. Target Label Creation
    print("--- 4. Creating Target Label 'fav_cover' ---")
    labeled_df = add_labels(merged_df)
    
    # 5. Feature Pruning: Remove Leakage Columns
    # Drop columns that directly reveal the outcome (scores, result, final margin)
    leak_cols = [
        "home_score", "away_score", "result", "total", "overtime",
        "game_type", "ats_margin_home" # The margin is the leakage column
    ]
    final_df = labeled_df.drop(columns=[c for c in leak_cols if c in labeled_df.columns], errors="ignore")
    
    # 6. Time-based Split: 2021-2024 train, 2025 test
    print("--- 5. Splitting Data (Train: 2021-2024, Test: 2025) ---")
    train_final = final_df[final_df["season"] < 2025].copy()
    test_final = final_df[final_df["season"] == 2025].copy()

    # 7. Saving Final Processed Data
    train_path = PROCESSED_DIR / "train_processed.csv"
    test_path = PROCESSED_DIR / "test_processed.csv"
    
    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)

    print(f"\n✅ Processing complete.")
    print(f"Training set size (2021-2024): {len(train_final)} games.")
    print(f"Test set size (2025):          {len(test_final)} games.")
    print(f"Saved files to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()