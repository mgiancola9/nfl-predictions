#!/usr/bin/env python3
"""
Join team strengths and injury features to games.
Creates opponent-context features (Deltas) and blends Priors for early season.

Logic:
1. Load Games, Rolling Stats, and Injury Data.
2. Create 'Priors' (Last year's final stats) to stabilize early season predictions.
3. Merge everything into the Game Schedule (Home vs Away).
4. Blend 'Current' vs 'Prior' stats based on the week number.
5. Calculate 'Delta' features (Home Advantage vs Away).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

def load_games() -> pd.DataFrame:
    path = DATA_RAW_DIR / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing games.csv at: {path}")
    
    games = pd.read_csv(path, low_memory=False)
    
    # FILTER: Regular Season AND Year >= 2021
    games = games[
        (games["game_type"] == "REG") & 
        (games["season"] >= 2021)  # <--- ADD THIS LINE
    ].copy()
    
    # Ensure standard columns exist
    req_cols = ['season', 'week', 'home_team', 'away_team', 'game_id']
    # If spread_line exists, keep it for reference, otherwise just keys
    if 'spread_line' in games.columns:
        req_cols.append('spread_line')
        
    return games[req_cols]

def load_rolling_stats() -> pd.DataFrame:
    path = DATA_INTERIM_DIR / "team_strengths_rolling.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing rolling stats: {path}. Run rolling_stats.py first.")
    return pd.read_csv(path)

def load_injury_features() -> pd.DataFrame:
    path = DATA_INTERIM_DIR / "injury_features.csv"
    if not path.exists():
        print(f"⚠️ Warning: Missing injury features: {path}. Proceeding without injuries.")
        return pd.DataFrame()
    return pd.read_csv(path)

def get_priors_from_rolling(rolling_df):
    """
    Extracts the final available stats from each season to serve as 
    'Prior Estimates' for the NEXT season.
    """
    # Sort by week to get the last entry for each team-season
    priors = rolling_df.sort_values(['season', 'team', 'week']).groupby(['season', 'team']).tail(1)
    
    # Shift Season Forward: 2024 Final Stats become 2025 Priors
    priors['season'] = priors['season'] + 1
    
    # Select feature columns (excluding week)
    feature_cols = [c for c in priors.columns if c not in ['season', 'week', 'team']]
    priors = priors[['season', 'team'] + feature_cols].copy()
    
    # Rename to indicate these are priors
    priors = priors.rename(columns={c: f"{c}_prior" for c in feature_cols})
    
    return priors

def merge_features(games, rolling, injuries, priors):
    print("   Merging Rolling Stats & Injuries...")
    
    # 1. Merge Rolling Stats (Home)
    df = games.merge(
        rolling.add_prefix("home_"),
        left_on=['season', 'week', 'home_team'],
        right_on=['home_season', 'home_week', 'home_team'],
        how='left'
    )
    
    # 2. Merge Rolling Stats (Away)
    df = df.merge(
        rolling.add_prefix("away_"),
        left_on=['season', 'week', 'away_team'],
        right_on=['away_season', 'away_week', 'away_team'],
        how='left'
    )
    
    # Drop redundant merge keys from rolling merge
    drop_cols = ['home_season', 'home_week', 'away_season', 'away_week']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 3. Merge Injury Stats (Home/Away)
    if not injuries.empty:
        inj_home = injuries.rename(columns={'team': 'home_team', 'vam_lost': 'home_vam_lost'})
        inj_away = injuries.rename(columns={'team': 'away_team', 'vam_lost': 'away_vam_lost'})
        
        df = df.merge(
            inj_home[['season', 'week', 'home_team', 'home_vam_lost']],
            on=['season', 'week', 'home_team'],
            how='left'
        )
        
        df = df.merge(
            inj_away[['season', 'week', 'away_team', 'away_vam_lost']],
            on=['season', 'week', 'away_team'],
            how='left'
        )
        
        df['home_vam_lost'] = df['home_vam_lost'].fillna(0)
        df['away_vam_lost'] = df['away_vam_lost'].fillna(0)
    else:
        df['home_vam_lost'] = 0
        df['away_vam_lost'] = 0

    # 4. Merge Priors
    print("   Merging Priors for Early Season Blending...")
    
    # FIX: Explicitly match 'season' to 'home_season' because of add_prefix
    df = df.merge(
        priors.add_prefix("home_"), 
        left_on=['season', 'home_team'], 
        right_on=['home_season', 'home_team'], 
        how='left'
    )
    
    df = df.merge(
        priors.add_prefix("away_"), 
        left_on=['season', 'away_team'], 
        right_on=['away_season', 'away_team'], 
        how='left'
    )
    
    # Cleanup redundant keys again (from Prior merge)
    df = df.drop(columns=['home_season', 'away_season'], errors='ignore')
    
    return df

def calculate_blended_features(df):
    print("   Calculating Blended Features (Current vs Prior)...")
    
    # Define Blend Factor alpha (0 to 1)
    # Week 1: 0.0 (100% Prior), Week 7+: 1.0 (100% Current)
    ramp_weeks = 6.0
    df['alpha'] = ((df['week'] - 1) / ramp_weeks).clip(0, 1)
    
    # Identify columns to blend
    features = [c for c in df.columns if c.endswith('_entering') and not c.endswith('_prior')]
    
    for col in features:
        prior_col = f"{col}_prior"
        
        if prior_col in df.columns:
            curr_val = df[col].fillna(df[prior_col])
            prior_val = df[prior_col].fillna(curr_val).fillna(0)
            
            df[f"{col}_blended"] = (curr_val * df['alpha']) + (prior_val * (1 - df['alpha']))
        else:
            df[f"{col}_blended"] = df[col].fillna(0)
            
    return df

def calculate_deltas(df):
    print("   Calculating Matchup Deltas...")
    
    # Net Rating (Offense - Defense)
    # Note: Defense Strength is 'Allowed EPA', so Lower is Better?
    # Let's check rolling_stats.py logic.
    # Typically: Offense Strength = EPA. Defense Strength = EPA Allowed.
    # So Net = Offense - Defense.
    
    df['home_net_rating'] = df['home_offense_strength_entering_blended'] - df['home_defense_strength_entering_blended']
    df['away_net_rating'] = df['away_offense_strength_entering_blended'] - df['away_defense_strength_entering_blended']
    
    df['delta_net_rating'] = df['home_net_rating'] - df['away_net_rating']
    
    # Matchup Advantages
    # Home Offense vs Away Defense
    df['home_matchup_adv'] = df['home_offense_strength_entering_blended'] - df['away_defense_strength_entering_blended']
    # Away Offense vs Home Defense
    df['away_matchup_adv'] = df['away_offense_strength_entering_blended'] - df['home_defense_strength_entering_blended']
    
    # Injury Delta (Positive = Home more injured)
    df['delta_vam_lost'] = df['home_vam_lost'] - df['away_vam_lost']
    
    return df

def main():
    print("--- Building Contextual Features ---")
    
    games = load_games()
    rolling = load_rolling_stats()
    injuries = load_injury_features()
    
    priors = get_priors_from_rolling(rolling)
    
    merged = merge_features(games, rolling, injuries, priors)
    blended = calculate_blended_features(merged)
    final_df = calculate_deltas(blended)
    
    out_path = DATA_INTERIM_DIR / "team_strengths_features.csv"
    final_df.to_csv(out_path, index=False)
    
    print(f"\n✅ Successfully built features: {out_path}")
    print(f"   Rows: {len(final_df)}")
    print(f"   Key Features: {[c for c in final_df.columns if 'delta' in c]}")

if __name__ == "__main__":
    main()