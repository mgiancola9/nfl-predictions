#!/usr/bin/env python3
"""
Final Data Assembly Pipeline.
UPDATED: Creates 'home_cover' target and standardizes spread perspective.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_games() -> pd.DataFrame:
    path = DATA_RAW_DIR / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing games.csv at: {path}")

    games = pd.read_csv(path, low_memory=False)

    # Filter to Regular Season and relevant years
    games = games[
        (games["game_type"] == "REG") &
        (games["season"] >= 2021)
    ].copy()
    
    # Filter out future games (those without scores)
    games = games.dropna(subset=['home_score', 'away_score'])
    
    cols_to_keep = [
        "game_id", "season", "week", "home_team", "away_team", 
        "home_score", "away_score", "spread_line", "total_line",
        "gameday", "weekday", "gametime", "location", "div_game", 
        "roof", "surface", "temp", "wind"
    ]
    
    existing_cols = [c for c in cols_to_keep if c in games.columns]
    return games[existing_cols]

def load_features() -> pd.DataFrame:
    path = DATA_INTERIM_DIR / "team_strengths_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}. Run build_features.py first.")
    return pd.read_csv(path)

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the betting target 'home_cover'.
    
    Logic:
    - spread_line is Away Perspective (e.g. -3.5 means Away is Favored).
    - We convert to Home Perspective for the calculation.
    - Home Spread = -1 * spread_line
    - ATS Margin = (Home Score - Away Score) + Home Spread
    
    Target:
    - 1 if Home Team Covered.
    - 0 if Home Team Did Not Cover (or Push).
    """
    df = df.copy()
    
    # Convert spread to Home perspective
    # e.g. Away -3.5 -> Home +3.5
    # e.g. Away +7.0 -> Home -7.0
    home_spread = -1 * df['spread_line']
    
    # Calculate Margin relative to spread
    df['ats_margin_home'] = (df['home_score'] - df['away_score']) + home_spread
    
    # Target: Did Home Cover?
    df['home_cover'] = (df['ats_margin_home'] > 0).astype(int)
    
    return df

def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leakage columns and prepares features."""
    
    # Create 'home_line' feature for clarity (Home Perspective Spread)
    # The model learns better if the spread aligns with the target (Home Cover)
    df['home_line'] = -1 * df['spread_line']
    
    leakage_cols = [
        "home_score", "away_score", "result", "total", "ats_margin_home",
        "spread_line" # We drop the raw away-line to avoid confusion
    ]
    
    return df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')

def main():
    print("--- Preprocessing Data Pipeline (Target: Home Cover) ---")
    
    games = load_games()
    features = load_features()
    
    print(f"   Loaded {len(games)} games and {len(features)} feature rows.")
    
    merged = games.merge(
        features,
        on=['season', 'week', 'home_team', 'away_team'],
        how='inner',
        suffixes=('', '_feat')
    )
    
    # Drop any redundant columns from features file (like spread_line duplicate)
    merged = merged.loc[:, ~merged.columns.str.endswith('_feat')]

    labeled = create_target(merged)
    final_df = finalize_columns(labeled)
    
    train_df = final_df[final_df['season'] < 2025].copy()
    test_df = final_df[final_df['season'] == 2025].copy()
    
    train_df.to_csv(DATA_PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test_processed.csv", index=False)
    
    print(f"\n Processing Complete.")
    print(f"   Training Data: {len(train_df)} rows")
    print(f"   Testing Data:  {len(test_df)} rows")

if __name__ == "__main__":
    main()