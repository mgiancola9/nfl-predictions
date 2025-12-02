import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def get_rolling_stats(stats_df, window=5):
    """
    Calculates rolling averages for team stats.
    Shift(1) ensures we only use PAST game data to predict the CURRENT game.
    """
    #Features columns to calculate rolling averages for
    cols_to_roll = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered', 'passing_first_downs',
        'passing_epa', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'penalty_yards', 
        'fg_made', 'fg_att', 'def_tackles_solo', 'def_tackle_assists'
    ]
    
    #Verify which columns actually exist in the dataframe before trying to roll them
    available_cols = [c for c in cols_to_roll if c in stats_df.columns]
    
    #Sort by team and game week to ensure correct rolling calculation
    stats_df = stats_df.sort_values(['team', 'season', 'week'])
    
    #Calculate Rolling Averages
    #group by team -> shift 1 row down (so week 5 sees week 4 data)
    rolling_stats = stats_df.groupby('team')[available_cols].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    #Rename columns to indicate they are averages
    rolling_stats.columns = [f'avg_{c}' for c in rolling_stats.columns]
    
    #Combine identifiers with the new stats
    result = pd.concat([stats_df[['season', 'week', 'team']], rolling_stats], axis=1)
    
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
        
    weekly_stats = pd.concat(all_stats_files, ignore_index=True)
    
    #Calculate Rolling Averages
    #print("Calculating rolling averages...")
    rolling_features = get_rolling_stats(weekly_stats, window=5)
    
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
    games = games.dropna(subset=['home_avg_passing_yards', 'away_avg_passing_yards'])
    
    return games


def preprocess(df, feature_columns, target_column, test_size = 0.2, scaler=None):
    """
    Preprocess merged NFL games DataFrame:
      - Select features and label
      - Train/validation split
      - Scale features
      - Convert to PyTorch tensors
    Returns: X_train_t, X_val_t, y_train_t, y_val_t, scaler
    """

    #Features and labels
    X = df[feature_columns].values
    y = df[target_column].values

    # 2. Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y)


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