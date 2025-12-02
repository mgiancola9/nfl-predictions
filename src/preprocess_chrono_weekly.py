import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def enrich_data(weekly_stats, games):
    """
    Extracts scores from the games file since stats file doesn't have them
    Calculates 'Yards Allowed' by finding the opponent's offensive stats
    """
    #Home scores perspective
    home_df = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']
    
    # Away scores perspective
    away_df = games[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']

    scores_df = pd.concat([home_df, away_df], ignore_index=True)

    #merge scores with weekly stats
    merged_df = weekly_stats.merge(scores_df, on=['season', 'week', 'team'], how='inner')

    #Now calculate 'yards_allowed' by merging opponent's offensive stats (their opffensive stats are this team's defensive stats)
    opp_cols = ['season', 'week', 'team', 'passing_yards', 'rushing_yards', 'passing_interceptions']
    opp_stats = merged_df[opp_cols].copy()
    opp_stats.columns = ['season', 'week', 'opponent', 'passing_yards_allowed', 'rushing_yards_allowed', 'def_interceptions_forced']

    final_df = merged_df.merge(opp_stats, on=['season', 'week', 'opponent'], how='left')
    final_df['total_yards_allowed'] = final_df['passing_yards_allowed'] + final_df['rushing_yards_allowed']

    return final_df

def get_rolling_stats(enriched_df, window=5):
    """
    Calculates rolling averages for team stats.
    Shift(1) ensures we only use PAST game data to predict the CURRENT game.
    """
    #Features columns to calculate rolling averages for
    cols_to_roll = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered', 'sack_fumbles', 'passing_first_downs',
        'passing_2pt_conversions', 'passing_epa', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'rushing_fumbles', 'rushing_first_downs', 
        'receiving_epa', 'fg_made', 'fg_att', 'penalty_yards', 'pat_made', 'pat_att','fumble_recovery_own', 'fumble_recovery_opp', 'points_scored', 'points_allowed', 
        'total_yards_allowed', 'passing_yards_allowed', 'rushing_yards_allowed', 'def_interceptions_forced', 'def_fumbles_forced']
    
    #Verify which columns actually exist in the dataframe before trying to roll them
    available_cols = [c for c in cols_to_roll if c in enriched_df.columns]
    
    #Sort by team and game week to ensure correct rolling calculation
    enriched_df = enriched_df.sort_values(['team', 'season', 'week'])
    
    #Calculate Rolling Averages
    #group by team -> shift 1 row down (so week 5 sees week 4 data)
    rolling_stats = enriched_df.groupby('team')[available_cols].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    #Rename columns to indicate they are averages
    rolling_stats.columns = [f'avg_{c}' for c in rolling_stats.columns]
    
    #Combine identifiers with the new stats
    result = pd.concat([enriched_df[['season', 'week', 'team']], rolling_stats], axis=1)
    
    return result

#Merge game data with team statistics for both home and away teams
def merge_data(file_name='games_train.csv'):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'Data')
    games = pd.read_csv(os.path.join(DATA_DIR, file_name))
    
    #Combine ALL Weekly Stats Files
    all_stats_files = []
    for year in range(2016, 2026):
        path = os.path.join(DATA_DIR, f'stats_team_week_{year}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_stats_files.append(df)
            
    if not all_stats_files:
        raise FileNotFoundError("No weekly stats files found!")
        
    raw_stats = pd.concat(all_stats_files, ignore_index=True)

    #Enrich stats with scores and opponent defensive data
    #print("Enriching stats with scores and opponent data...")
    enriched_stats = enrich_data(raw_stats, games)
    
    #Calculate Rolling Averages
    #print("Calculating rolling averages...")
    rolling_features = get_rolling_stats(enriched_stats, window=5)
    
    #Merge Home Team Stats
    #Match Game's (Season, Week, HomeTeam) with Stats' (Season, Week, Team)
    #print("Merging home team data...")
    games = games.merge(
        rolling_features.add_prefix('home_'),
        left_on=['season', 'week', 'home_team'],
        right_on=['home_season', 'home_week', 'home_team'],
        how='left'
    )
    
    #Merge Away Team Stats
    #print("Merging away team data...")
    games = games.merge(
        rolling_features.add_prefix('away_'),
        left_on=['season', 'week', 'away_team'],
        right_on=['away_season', 'away_week', 'away_team'],
        how='left'
    )
    
    #Drop rows where we don't have stats (Week 1 of 2016))
    games = games.dropna(subset=['home_avg_points_scored'])
    
    return games


def preprocess(train_df, test_df, feature_columns, target_column, scaler=None):
    """
    Preprocess merged NFL games DataFrame:
      - Select features and label
      - Train/validation split
      - Scale features
      - Convert to PyTorch tensors
    Returns: X_train_t, X_val_t, y_train_t, y_val_t, scaler

    Train set: 2016-2023
    Test set: 2024-2025 week 13
    Train in chronological order to avoid data leakage.
    """

    #Features and labels
    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values

    X_val = test_df[feature_columns].values
    y_val = test_df[target_column].values

    #Scale features
    if scaler is None: #Initialize new scaler if not provided
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train) #Use provided scaler from reloaded model
    
    X_val_scaled = scaler.transform(X_val)

    #Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    return X_train_t, X_val_t, y_train_t, y_val_t, scaler

if __name__ == "__main__":
    merged_train_data = merge_data("games_train.csv")
    merged_train_data.head(100).to_csv("merged_train_data_sample.csv", index=False)
    #print("Data merged and preprocessed.")
                       
