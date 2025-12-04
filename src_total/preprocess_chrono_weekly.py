import os
import pandas as pd
import numpy as np

def enrich_data(weekly_stats, games):
    """
    Extracts scores from the games file since stats file doesn't have them
    Calculates 'Yards Allowed' by finding the opponent's offensive stats
    """
    # Home scores perspective
    home_df = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']
    
    # Away scores perspective
    away_df = games[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']

    scores_df = pd.concat([home_df, away_df], ignore_index=True)

    # Merge scores with weekly stats
    merged_df = weekly_stats.merge(scores_df, on=['season', 'week', 'team'], how='inner')

    # Now calculate 'yards_allowed' by merging opponent's offensive stats
    opp_cols = ['season', 'week', 'team', 'passing_yards', 'rushing_yards', 'passing_interceptions', 'passing_epa', 'rushing_epa']
    opp_stats = merged_df[opp_cols].copy()
    opp_stats.columns = ['season', 'week', 'opponent', 'passing_yards_allowed', 'rushing_yards_allowed', 
                         'def_interceptions_forced', 'passing_epa_allowed', 'rushing_epa_allowed']

    final_df = merged_df.merge(opp_stats, on=['season', 'week', 'opponent'], how='left')
    final_df['total_yards_allowed'] = final_df['passing_yards_allowed'] + final_df['rushing_yards_allowed']

    return final_df

def get_rolling_stats(enriched_df, window=5):
    """
    Calculates rolling averages for team stats.
    Shift(1) ensures we only use PAST game data to predict the CURRENT game.
    """
    cols_to_roll = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered', 'sack_fumbles', 'passing_first_downs',
        'passing_2pt_conversions', 'passing_epa', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'rushing_fumbles', 'rushing_first_downs', 
        'receiving_epa', 'fg_made', 'fg_att', 'penalty_yards', 'pat_made', 'pat_att','fumble_recovery_own', 'fumble_recovery_opp', 'points_scored', 'points_allowed', 
        'total_yards_allowed', 'passing_yards_allowed', 'rushing_yards_allowed', 'def_interceptions_forced', 'passing_epa_allowed', 'rushing_epa_allowed']
    
    available_cols = [c for c in cols_to_roll if c in enriched_df.columns]
    
    # Sort to ensure correct rolling calculation
    enriched_df['week'] = enriched_df['week'].astype(int)
    enriched_df = enriched_df.sort_values(['team', 'season', 'week'])
    
    # Calculate Rolling Averages (Group -> Shift -> Roll)
    rolling_stats = enriched_df.groupby('team')[available_cols].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    rolling_stats.columns = [f'avg_{c}' for c in rolling_stats.columns]
    
    result = pd.concat([enriched_df[['season', 'week', 'team']], rolling_stats], axis=1)
    return result

def merge_data(file_name='games_train.csv'):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'Data_Total')
    games = pd.read_csv(os.path.join(DATA_DIR, file_name))
    
    all_stats_files = []
    # Adjusted range to include 2025 if available
    for year in range(2016, 2026):
        path = os.path.join(DATA_DIR, f'stats_team_week_{year}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_stats_files.append(df)
            
    if not all_stats_files:
        raise FileNotFoundError("No weekly stats files found!")
        
    raw_stats = pd.concat(all_stats_files, ignore_index=True)

    enriched_stats = enrich_data(raw_stats, games)
    rolling_features = get_rolling_stats(enriched_stats, window=5)
    
    # Merge Home Team Stats
    games = games.merge(
        rolling_features.add_prefix('home_'),
        left_on=['season', 'week', 'home_team'],
        right_on=['home_season', 'home_week', 'home_team'],
        how='left'
    )
    
    # Merge Away Team Stats
    games = games.merge(
        rolling_features.add_prefix('away_'),
        left_on=['season', 'week', 'away_team'],
        right_on=['away_season', 'away_week', 'away_team'],
        how='left'
    )
    
    # Drop rows where we don't have stats (Week 1 of 2016)
    games = games.dropna(subset=['home_avg_points_scored'])
    
    return games

def get_features_and_labels(df, feature_columns, target_column):
    """
    Simple extractor. No scaling, no tensors.
    """
    X = df[feature_columns].values
    y = df[target_column].values
    return X, y