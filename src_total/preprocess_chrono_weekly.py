import os
import pandas as pd
import numpy as np

def enrich_data(weekly_stats, games):
    """
    Extracts scores and calculates 'Allowed' metrics (Defense)
    """
    # 1. SCOREBOARD MERGE
    # Home perspective
    home_df = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']
    
    # Away perspective
    away_df = games[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']

    scores_df = pd.concat([home_df, away_df], ignore_index=True)
    merged_df = weekly_stats.merge(scores_df, on=['season', 'week', 'team'], how='right')

    # 2. CALCULATE OPPONENT STATS ("Allowed")
    # We grab the stats of the *opponent* to see what the *team* allowed.
    # Added 'plays' to this list so we track 'plays_allowed'
    opp_cols = [
        'season', 'week', 'team', 
        'passing_yards', 'rushing_yards', 'passing_interceptions', 
        'passing_epa', 'rushing_epa', 'plays'  # <--- NEW
    ]
    
    opp_stats = merged_df[opp_cols].copy()
    
    # Rename to '_allowed'
    opp_stats.columns = [
        'season', 'week', 'opponent', 
        'passing_yards_allowed', 'rushing_yards_allowed', 'def_interceptions_forced', 
        'passing_epa_allowed', 'rushing_epa_allowed', 'plays_allowed' # <--- NEW
    ]

    final_df = merged_df.merge(opp_stats, on=['season', 'week', 'opponent'], how='left')
    final_df['total_yards_allowed'] = final_df['passing_yards_allowed'] + final_df['rushing_yards_allowed']

    return final_df

def get_rolling_stats(enriched_df, window=5):
    """
    Calculates rolling averages.
    """
    cols_to_roll = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks',
        'passing_first_downs', 'passing_2pt_conversions', 'passing_epa', 
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'rushing_fumbles', 'rushing_first_downs', 
        'receiving_epa', 'fg_made', 'fg_att', 'penalty_yards', 
        'points_scored', 'points_allowed', 'total_yards_allowed', 
        'passing_yards_allowed', 'rushing_yards_allowed', 'def_interceptions_forced', 
        'passing_epa_allowed', 'rushing_epa_allowed',
        'plays', 'plays_allowed' # <--- NEW: Roll the pace metrics
    ]
    
    # Filter for columns that actually exist
    available_cols = [c for c in cols_to_roll if c in enriched_df.columns]
    
    # Sort for rolling
    enriched_df['week'] = enriched_df['week'].astype(int)
    enriched_df = enriched_df.sort_values(['team', 'season', 'week'])
    
    # Group -> Shift(1) -> Rolling Mean
    # shift(1) guarantees we only use PAST data for current prediction
    rolling_stats = enriched_df.groupby('team')[available_cols].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    rolling_stats.columns = [f'avg_{c}' for c in rolling_stats.columns]
    
    result = pd.concat([enriched_df[['season', 'week', 'team']], rolling_stats], axis=1)
    return result

def merge_data(file_name='games_2016_2025.csv'):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'Data_Total')
    games = pd.read_csv(os.path.join(DATA_DIR, file_name))
    
    all_stats_files = []
    # Load all stats files
    for year in range(2016, 2026):
        path = os.path.join(DATA_DIR, f'stats_team_week_{year}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_stats_files.append(df)
            
    if not all_stats_files:
        raise FileNotFoundError("No weekly stats files found!")
        
    raw_stats = pd.concat(all_stats_files, ignore_index=True)

    # --- FEATURE ENGINEERING: CALCULATE PACE ---
    # Plays = Pass Attempts + Rush Attempts + Sacks
    # Handle naming variations (sacks vs sacks_suffered)
    if 'sacks_suffered' in raw_stats.columns:
        sacks_col = 'sacks_suffered'
    else:
        sacks_col = 'sacks'
        
    # Fill NaNs with 0 to avoid breaking the sum
    raw_stats[sacks_col] = raw_stats[sacks_col].fillna(0)
    raw_stats['attempts'] = raw_stats['attempts'].fillna(0)
    raw_stats['carries'] = raw_stats['carries'].fillna(0)

    # Create the 'plays' feature
    raw_stats['plays'] = raw_stats['attempts'] + raw_stats['carries'] + raw_stats[sacks_col]
    
    # Rename sack column for consistency if needed
    if sacks_col != 'sacks':
        raw_stats['sacks'] = raw_stats[sacks_col]

    # Enrich and Roll
    enriched_stats = enrich_data(raw_stats, games)
    rolling_features = get_rolling_stats(enriched_stats, window=5)
    
    # Merge Home Stats
    games = games.merge(
        rolling_features.add_prefix('home_'),
        left_on=['season', 'week', 'home_team'],
        right_on=['home_season', 'home_week', 'home_team'],
        how='left'
    )
    
    # Merge Away Stats
    games = games.merge(
        rolling_features.add_prefix('away_'),
        left_on=['season', 'week', 'away_team'],
        right_on=['away_season', 'away_week', 'away_team'],
        how='left'
    )
    
    # Drop rows missing stats (Week 1 2016)
    games = games.dropna(subset=['home_avg_points_scored'])
    
    return games

def get_features_and_labels(df, feature_columns, target_column):
    X = df[feature_columns].values
    y = df[target_column].values
    return X, y