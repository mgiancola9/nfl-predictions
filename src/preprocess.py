import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

#Merge game data with team statistics for both home and away teams
def merge_data():
    '''
    Returns a DataFrame merging game data with team statistics for both home and away teams.
    '''

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'Data')
    games = pd.read_csv(os.path.join(DATA_DIR, 'games.csv'))

    team_stats = pd.concat(
        [pd.read_csv(os.path.join(DATA_DIR, f'stats_team_reg_{year}.csv')) for year in range(2021, 2026)],
        ignore_index=True
    )

    #Prepare away team stats
    away_stats = team_stats.copy()
    away_stats = away_stats.add_prefix('away_')
    away_stats.rename(columns={'away_season': 'season', 'away_team': 'away_team'}, inplace=True)

    #Merge away team stats
    games = games.merge(
        away_stats,
        on=['season', 'away_team'],
        how='left'
    )

    #Prepare home team stats
    home_stats = team_stats.copy()
    home_stats = home_stats.add_prefix('home_')
    home_stats.rename(columns={'home_season': 'season', 'home_team': 'home_team'}, inplace=True)

    #Merge home team stats
    games = games.merge(
        home_stats,
        on=['season', 'home_team'],
        how='left'
    )

    #convert categorical features to numbers using one-hot encoding
    games['roof'] = games['roof'].fillna('unknown').replace('', 'unknown')
    games['surface'] = games['surface'].fillna('unknown').replace('', 'unknown')
    games = pd.get_dummies(games, columns=['roof', 'surface'])

    #games.head(10).to_csv('merged_data_sample.csv', index=False)

    return games


def preprocess(df, feature_columns, target_column, test_size = 0.2):
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    #Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    return X_train_t, X_val_t, y_train_t, y_val_t, scaler

