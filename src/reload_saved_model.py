import preprocess
import NN_dropout
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import joblib

merged_data = preprocess.merge_data("games_train.csv")

feature_columns = ['total_line','over_odds','under_odds','spread_line','away_moneyline','home_moneyline',
'away_spread_odds','home_spread_odds','week','temp','wind','away_rest','home_rest', 'season',
#Away team stats
'away_passing_yards','away_passing_tds','away_passing_epa','away_rushing_yards','away_rushing_tds','away_rushing_epa',
'away_receiving_yards','away_receiving_tds','away_receiving_epa','away_def_sacks','away_def_interceptions',
#Home team stats
'home_passing_yards','home_passing_tds','home_passing_epa','home_rushing_yards','home_rushing_tds','home_rushing_epa',
'home_receiving_yards','home_receiving_tds','home_receiving_epa','home_def_sacks','home_def_interceptions',
]

target_col = 'over_hit'

scaler = joblib.load("Enter Scaler Path Here")  # Load the scaler used during model training

X_train_t, X_val_t, y_train_t, y_val_t, scaler = preprocess.preprocess(merged_data, feature_columns, target_col)

input_features = X_train_t.shape[1]

DROPOUT_RATE = 0.5
MODEL_PATH = "Enter Model Path Here"

model = NN_dropout.FeedForwardNetWithDropout(input_size=input_features, dropout_rate=DROPOUT_RATE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

criterion = torch.nn.BCELoss()
with torch.no_grad():
    train_outputs = model(X_train_t)
    val_outputs = model(X_val_t)
    train_loss = criterion(train_outputs, y_train_t)
    val_loss = criterion(val_outputs, y_val_t)

print(f"Reloaded Model from {MODEL_PATH}")
print(f"Train Loss: {train_loss.item():.4f}")
print(f"Validation Loss: {val_loss.item():.4f}")




