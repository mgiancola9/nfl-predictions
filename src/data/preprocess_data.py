#!/usr/bin/env python3
"""
Final Data Assembly Pipeline.

1. Loads the raw schedule/outcomes (games.csv).
2. Loads the engineered features (team_strengths_features.csv).
3. Merges them into a single Master DataFrame.
4. Creates the Target Label ('fav_cover').
5. Removes "Future Data" (Scores, Results) to prevent leakage.
6. Splits into Train (2021-2024) and Test (2025).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure output directory exists
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_games() -> pd.DataFrame:
    """Load raw game schedule and outcomes."""
    path = DATA_RAW_DIR / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing games.csv at: {path}")

    games = pd.read_csv(path, low_memory=False)

    # Filter to Regular Season and relevant years
    games = games[
        (games["game_type"] == "REG") &
        (games["season"] >= 2021)
    ].copy()
    
    # --- ADD THIS LINE ---
    # Remove games that haven't been played yet (no score)
    games = games.dropna(subset=['home_score', 'away_score']) 
    
    # Standardize columns
    cols_to_keep = [
        "game_id", "season", "week", "home_team", "away_team", 
        "home_score", "away_score", "spread_line", "total_line",
        "gameday", "weekday", "gametime", "location", "div_game", 
        "roof", "surface", "temp", "wind"
    ]
    
    existing_cols = [c for c in cols_to_keep if c in games.columns]
    return games[existing_cols]

def load_features() -> pd.DataFrame:
    """Load the features created by build_features.py."""
    path = DATA_INTERIM_DIR / "team_strengths_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}. Run build_features.py first.")
    
    return pd.read_csv(path)

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the betting target 'fav_cover'.
    
    Logic:
    - Spread is from AWAY perspective (e.g., -3.5 means Away is favored).
    - Calculate the Home Team's Margin (Home - Away).
    - Calculate ATS Margin = Home Margin + Spread.
    - If ATS Margin > 0, Home Covered.
    - If ATS Margin < 0, Away Covered.
    
    Target 'fav_cover':
    - 1 if the Favorite covered.
    - 0 if the Underdog covered.
    """
    df = df.copy()
    
    # 1. Determine who is the favorite
    # spread_line < 0 -> Away Fav
    # spread_line > 0 -> Home Fav
    df['home_is_fav'] = df['spread_line'] > 0
    df['away_is_fav'] = df['spread_line'] < 0
    
    # 2. Calculate Home ATS Margin
    # Example: Home Score 24, Away 20. Spread -3 (Away favored by 3).
    # Home Margin = 4.
    # ATS Margin = 4 + (-3) = +1. Home Covered.
    df['ats_margin_home'] = (df['home_score'] - df['away_score']) + df['spread_line']
    
    # 3. Determine Result
    # Did Home Cover? (ATS Margin > 0)
    home_covered = df['ats_margin_home'] > 0
    away_covered = df['ats_margin_home'] < 0
    
    # 4. Set Target
    # If Home is Fav AND Home Covered -> 1
    # If Away is Fav AND Away Covered -> 1
    # Else -> 0
    df['fav_cover'] = 0
    
    cover_condition = (
        (df['home_is_fav'] & home_covered) |
        (df['away_is_fav'] & away_covered)
    )
    
    df.loc[cover_condition, 'fav_cover'] = 1
    
    # Handle Pushes (ATS Margin == 0)
    # We can drop them or keep them as 0. For binary class, dropping is cleaner.
    # But for now, we'll leave them as 0 (Loss for the bettor).
    
    return df

def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leakage columns (scores) and prepares final set."""
    
    # Drop outcome columns that would leak the result to the model
    leakage_cols = [
        "home_score", "away_score", "result", "total", "ats_margin_home",
        "home_is_fav", "away_is_fav"
    ]
    
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')
    
    return df

def main():
    print("--- Preprocessing Data Pipeline ---")
    
    # 1. Load
    games = load_games()
    features = load_features()
    
    print(f"   Loaded {len(games)} games and {len(features)} feature rows.")
    
    # 2. Merge
    # We use 'inner' merge to ensure we only keep games we have features for
    # (and removes games with no matching IDs)
    merged = games.merge(
        features,
        on=['season', 'week', 'home_team', 'away_team'],
        how='inner',
        suffixes=('', '_feat')
    )
    
    # Drop redundant columns from merge if any
    merged = merged.loc[:, ~merged.columns.str.endswith('_feat')]

    # 3. Create Target
    # We need scores to create the target, so we do this BEFORE dropping leakage
    labeled = create_target(merged)
    
    # 4. Clean up
    final_df = finalize_columns(labeled)
    
    # 5. Split (2021-2024 vs 2025)
    train_df = final_df[final_df['season'] < 2025].copy()
    test_df = final_df[final_df['season'] == 2025].copy()
    
    # 6. Save
    train_df.to_csv(DATA_PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test_processed.csv", index=False)
    
    print(f"\n✅ Processing Complete.")
    print(f"   Training Data: {len(train_df)} rows (2021-2024)")
    print(f"   Testing Data:  {len(test_df)} rows (2025)")
    print(f"   Files saved to: {DATA_PROCESSED_DIR}")

if __name__ == "__main__":
    main()