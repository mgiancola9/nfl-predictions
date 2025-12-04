#!/usr/bin/env python3
"""
Train the Spread Model using Walk-Forward Validation (Expanding Window).

Strategy:
1. Load 'train_processed.csv' (2021-2024 History).
2. Load 'test_processed.csv' (2025 Season).
3. Iterate through each week of 2025:
   - Train on History + All 2025 games played BEFORE that week.
   - Predict the games for the CURRENT week.
   - Store predictions and move to the next week.
4. Calculate final profitability metrics (Accuracy, Edge).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Columns to exclude from training features (Metadata + Leaks)
# NOTE: We dropped 'spread_line' in preprocessing and created 'home_line'.
# We want to KEEP 'home_line' as a feature, so do not add it here.
DROP_COLS = [
    "season", "week", "game_id", "home_team", "away_team", 
    "gameday", "weekday", "gametime", "home_cover", "fav_cover", # Targets
    "total_line", "location", "div_game", 
    "roof", "surface", "temp", "wind" 
]

def load_data():
    """Load the processed training and test sets."""
    train_path = DATA_PROCESSED_DIR / "train_processed.csv"
    test_path = DATA_PROCESSED_DIR / "test_processed.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed data not found in {DATA_PROCESSED_DIR}. Run preprocess_data.py first.")

    print(f"Loading data from {DATA_PROCESSED_DIR}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def get_features_and_target(df):
    """Separates X (features) and y (target)."""
    # Identify feature columns (everything except DROP_COLS)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    
    X = df[feature_cols].copy()
    # TARGET IS NOW 'home_cover'
    y = df["home_cover"].copy()
    
    return X, y

def train_predict_walk_forward(train_base, test_season):
    """
    Performs the expanding window validation.
    """
    # Sort test season by week to simulate chronological time
    test_season = test_season.sort_values("week")
    weeks = sorted(test_season["week"].unique())
    
    all_predictions = []
    
    print(f"\n--- Starting Walk-Forward Validation (Weeks {weeks[0]}-{weeks[-1]}) ---")
    
    for current_week in weeks:
        # 1. Define the Training Window
        #    Train = Base History (2021-2024) + 2025 games *before* this week
        current_season_history = test_season[test_season["week"] < current_week]
        train_data = pd.concat([train_base, current_season_history], axis=0)
        
        # 2. Define the Test Window (Only the current week)
        test_data = test_season[test_season["week"] == current_week].copy()
        
        if test_data.empty:
            continue
            
        # 3. Prepare Features (X) and Target (y)
        X_train, y_train = get_features_and_target(train_data)
        X_test, _ = get_features_and_target(test_data)
        
        # 4. Train Model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # 5. Predict
        #    [0] is prob of Home NOT Covering, [1] is prob of Home Covering
        probs = model.predict_proba(X_test)[:, 1]
        
        # 6. Store Predictions
        test_data["pred_prob_home"] = probs
        
        # Decision Logic:
        # If Prob(Home Cover) > 0.50 -> Predict 1 (Home Cover)
        # Else -> Predict 0 (Away Cover)
        test_data["pred_pick"] = (probs >= 0.50).astype(int)
        
        # Edge Calculation
        test_data["edge"] = np.abs(test_data["pred_prob_home"] - 0.5)
        
        all_predictions.append(test_data)
        
        print(f"   Week {current_week}: Trained on {len(train_data)} games.")

    if not all_predictions:
        return pd.DataFrame()
        
    return pd.concat(all_predictions, axis=0)

def evaluate_performance(results_df):
    """Calculates accuracy and betting metrics."""
    if results_df.empty:
        print("No predictions were made.")
        return

    y_true = results_df["home_cover"]
    y_pred = results_df["pred_pick"]
    
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, results_df["pred_prob_home"])
    
    print("\n" + "="*40)
    print(f"RESULTS: 2025 Walk-Forward (Target: Home Cover)")
    print("="*40)
    print(f"Total Games Predicted: {len(results_df)}")
    print(f"Overall Accuracy:      {acc:.2%}")
    print(f"Brier Score:           {brier:.4f}")
    print("-" * 40)
    
    # Profitability Check at different confidence levels
    thresholds = [0.50, 0.525, 0.55]
    
    for t in thresholds:
        # Filter for games where model confidence is above threshold
        min_edge = t - 0.5
        confident_picks = results_df[results_df["edge"] >= min_edge]
        
        if len(confident_picks) > 0:
            conf_acc = accuracy_score(confident_picks["home_cover"], confident_picks["pred_pick"])
            print(f"Accuracy at >{t:.1%} confidence ({len(confident_picks)} games): {conf_acc:.2%}")
        else:
            print(f"Accuracy at >{t:.1%} confidence: No games found")

def main():
    # 1. Load Data
    train_base, test_season = load_data()
    
    # 2. Run Simulation
    results = train_predict_walk_forward(train_base, test_season)
    
    # 3. Evaluate
    evaluate_performance(results)
    
    # 4. Save detailed log for review
    output_file = DATA_PROCESSED_DIR / "predictions_2025.csv"
    results.to_csv(output_file, index=False)
    print(f"\nDetailed predictions saved to: {output_file}")

if __name__ == "__main__":
    main()