#!/usr/bin/env python3
"""
Calculate ROLLING (Expanding Window) stats for teams.

Features:
- Offensive Strength: Calculated from team's own EPA/Turnovers.
- Defensive Strength: Calculated by 'flipping' the field (finding what opponents did against them).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def load_local_files(start_year, end_year):
    """Loads local stats_team_week_{year}.csv files."""
    all_frames = []
    
    for year in range(start_year, end_year + 1):
        possible_names = [f"stats_team_week_{year}.csv", f"stats_team_week_{year} (1).csv"]
        file_path = None
        for name in possible_names:
            p = DATA_RAW_DIR / name
            if p.exists():
                file_path = p
                break
        
        if file_path:
            print(f"   ✅ Loading {year}: {file_path.name}")
            df = pd.read_csv(file_path)
            df = process_team_file(df)
            all_frames.append(df)
        else:
            print(f"   ⚠️ Warning: No file found for {year}. Skipping.")
            
    if not all_frames:
        raise FileNotFoundError("No weekly team files were loaded!")
        
    # Combine and drop potential duplicates
    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['season', 'week', 'team'])
    return full_df

def process_team_file(df):
    """Normalizes columns."""
    df = df.copy()
    if 'recent_team' in df.columns:
        df = df.rename(columns={'recent_team': 'team'})
        
    # Ensure composite stats exist
    if 'turnovers' not in df.columns:
        tov_cols = ['passing_interceptions', 'rushing_fumbles_lost', 'sack_fumbles_lost', 'receiving_fumbles_lost']
        df['turnovers'] = 0
        for c in tov_cols:
            if c in df.columns: df['turnovers'] += df[c].fillna(0)
            
    if 'sacks_suffered' not in df.columns and 'sacks' in df.columns:
        df = df.rename(columns={'sacks': 'sacks_suffered'})

    # Keep necessary columns (Added opponent_team for the defensive flip)
    req_cols = ['season', 'week', 'team', 'opponent_team', 'passing_epa', 'rushing_epa', 'turnovers', 'sacks_suffered']
    for c in req_cols:
        if c not in df.columns: df[c] = 0.0
            
    return df[req_cols]

def calculate_rolling_windows(df):
    """
    Calculates expanding window stats for Offense AND Defense (Opponent EPA).
    """
    print("   Calculating expanding window stats (Offense & Defense)...")
    
    # --- 1. GET DEFENSIVE STATS (Flip the Field) ---
    # We want to know: "How much EPA does this team ALLOW?"
    # Answer: The EPA their opponent generated in the game.
    
    # Create a lookup for opponent stats
    # We look for the game where 'team' matches our 'opponent_team'
    opponent_stats = df[['season', 'week', 'team', 'passing_epa', 'rushing_epa']].copy()
    opponent_stats = opponent_stats.rename(columns={
        'team': 'opponent_team',       # Join key
        'passing_epa': 'def_passing_epa_allowed',
        'rushing_epa': 'def_rushing_epa_allowed'
    })
    
    # Merge defensive stats onto the main dataframe
    df = df.merge(
        opponent_stats,
        on=['season', 'week', 'opponent_team'],
        how='left'
    )
    
    # --- 2. CALCULATE ROLLING AVERAGES ---
    df = df.sort_values(['season', 'team', 'week'])
    
    # Metrics to roll: Offense (generated) and Defense (allowed)
    metrics = [
        'passing_epa', 'rushing_epa', 'turnovers',          # Offense
        'def_passing_epa_allowed', 'def_rushing_epa_allowed' # Defense
    ]
    
    for m in metrics:
        col_name = f'avg_{m}_entering'
        # Group by Season & Team -> Expanding Mean -> Shift 1 (Leak Prevention)
        df[col_name] = df.groupby(['season', 'team'])[m]\
                         .transform(lambda x: x.expanding().mean().shift(1))

    # --- 3. COMPOSITE STRENGTH SCORES ---
    
    # Offense Strength (Higher is Better)
    df['offense_strength_entering'] = (
        df['avg_passing_epa_entering'].fillna(0) + 
        df['avg_rushing_epa_entering'].fillna(0)
    )
    
    # Defense Strength (Higher is WORSE/Weaker because it's EPA Allowed)
    # The model will learn this direction automatically.
    df['defense_strength_entering'] = (
        df['avg_def_passing_epa_allowed_entering'].fillna(0) + 
        df['avg_def_rushing_epa_allowed_entering'].fillna(0)
    )
    
    # Clean up: Drop the raw game stats (leaks), keep only entering stats
    keep_cols = ['season', 'week', 'team', 'offense_strength_entering', 'defense_strength_entering'] + \
                [f'avg_{m}_entering' for m in metrics]
    
    return df[keep_cols]

def main():
    print("--- Generating Rolling Stats (Offense + Defense) ---")
    full_df = load_local_files(2021, 2025)
    rolling_df = calculate_rolling_windows(full_df)
    
    output_path = DATA_INTERIM_DIR / "team_strengths_rolling.csv"
    rolling_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved rolling stats to: {output_path}")
    print(f"   Rows: {len(rolling_df)}")
    print(f"   Sample (Defense Columns):")
    print(rolling_df[['season', 'week', 'team', 'defense_strength_entering']].dropna().head(3))

if __name__ == "__main__":
    main()