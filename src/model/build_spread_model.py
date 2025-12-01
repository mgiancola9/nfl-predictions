#!/usr/bin/env python3
"""
Train the Spread Model using Walk-Forward Validation (Expanding Window).

Strategy:
1. Train on historical data (2021-2024).
2. For the 2025 test season, iterate week-by-week.
3. For each week, retrain the model including ALL data prior to that week.
4. Predict the current week and store results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Columns to exclude from training features (Metadata + Leaks)
# Note: 'fav_cover' is our target.
DROP_COLS = [
    "season", "week", "game_id", "home_team", "away_team", 
    "gameday", "weekday", "gametime", "fav_cover",
    # Add any other metadata columns present in your csv
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
    y = df["fav_cover"].copy()
    
    return X, y

def train_predict_walk_forward(train_base, test_season):
    """
    Performs the expanding window validation.
    
    Args:
        train_base (pd.DataFrame): The static history (2021-2024).
        test_season (pd.DataFrame): The current season to simulate (2025).
        
    Returns:
        pd.DataFrame: The test_season dataframe with added 'pred_prob' and 'pred_class' columns.
    """
    # Sort test season by week to simulate chronological time
    test_season = test_season.sort_values("week")
    weeks = sorted(test_season["week"].unique())
    
    all_predictions = []
    
    print(f"\n--- Starting Walk-Forward Validation (Weeks {weeks[0]}-{weeks[-1]}) ---")
    
    for current_week in weeks:
        # 1. Define the Training Window
        #    Train = Base History + 2025 games *before* this week
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
        #    Using standard XGBoost parameters - tune these later!
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 5. Predict
        #    [0] is prob of class 0, [1] is prob of class 1 (Favorite Covers)
        probs = model.predict_proba(X_test)[:, 1]
        
        # 6. Store Predictions
        test_data["pred_prob"] = probs
        test_data["pred_class"] = (probs >= 0.5).astype(int)
        
        # Optional: Calculate 'Edge' (Confidence - 50%)
        test_data["edge"] = np.abs(test_data["pred_prob"] - 0.5)
        
        all_predictions.append(test_data)
        
        print(f"   Week {current_week}: Trained on {len(train_data)} games. Predicted {len(test_data)} games.")

    return pd.concat(all_predictions, axis=0)

def evaluate_performance(results_df):
    """Calculates accuracy and betting metrics."""
    y_true = results_df["fav_cover"]
    y_pred = results_df["pred_class"]
    
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, results_df["pred_prob"])
    
    print("\n" + "="*40)
    print(f"RESULTS: 2025 Walk-Forward Validation")
    print("="*40)
    print(f"Total Games Predicted: {len(results_df)}")
    print(f"Overall Accuracy:      {acc:.2%}")
    print(f"Brier Score:           {brier:.4f}")
    print("-" * 40)
    
    # Check profitability on "High Confidence" picks
    # (e.g. model implies >53% or >55% probability)
    thresholds = [0.53, 0.55, 0.57]
    for t in thresholds:
        # Filter for games where prob > t OR prob < (1-t)
        # i.e., model is confident either Yes or No
        confident_picks = results_df[results_df["edge"] >= (t - 0.5)]
        
        if len(confident_picks) > 0:
            conf_acc = accuracy_score(confident_picks["fav_cover"], confident_picks["pred_class"])
            print(f"Accuracy at >{t:.0%} confidence ({len(confident_picks)} games): {conf_acc:.2%}")
        else:
            print(f"Accuracy at >{t:.0%} confidence: No games met criteria")

def main():
    # 1. Load
    train_base, test_season = load_data()
    
    # 2. Walk-Forward Train & Predict
    results = train_predict_walk_forward(train_base, test_season)
    
    # 3. Evaluate
    evaluate_performance(results)
    
    # 4. Save detailed predictions for review
    output_file = DATA_PROCESSED_DIR / "predictions_2025_walk_forward.csv"
    results.to_csv(output_file, index=False)
    print(f"\nDetailed predictions saved to: {output_file}")

if __name__ == "__main__":
    main()