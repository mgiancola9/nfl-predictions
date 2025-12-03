import xgboost as xgb
import preprocess_chrono_weekly as preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_xgboost_regression():
    # 1. Load Data
    print("Loading data...")
    merged_train_data = preprocess.merge_data("games_train.csv")
    merged_test_data = preprocess.merge_data("games_test.csv")

    # 2. Create Regression Target
    # We want to predict the total points scored
    merged_train_data['actual_total'] = merged_train_data['home_score'] + merged_train_data['away_score']
    merged_test_data['actual_total'] = merged_test_data['home_score'] + merged_test_data['away_score']
    
    target_col = 'actual_total'

    # 3. Define Features (Keep the "Hatchet" list + EPA)
    feature_columns = [
        'total_line', 'spread_line', 'week', 'temp', 'wind', 'away_rest', 'home_rest',
        'away_avg_passing_epa', 'home_avg_passing_epa_allowed',
        'away_avg_rushing_epa', 'home_avg_rushing_epa_allowed',
        'home_avg_passing_epa', 'away_avg_passing_epa_allowed',
        'home_avg_rushing_epa', 'away_avg_rushing_epa_allowed',
        'away_avg_points_scored', 'home_avg_points_allowed',
        'home_avg_points_scored', 'away_avg_points_allowed'
    ]

    # 4. Preprocess 
    # Note: We pass target_col, but we'll swap it manually for the regression target
    print("Preprocessing...")
    # We cheat slightly here to use the existing function, but we extract the numpy arrays directly
    X_train_t, X_val_t, _, _, scaler = preprocess.preprocess(
        merged_train_data, 
        merged_test_data, 
        feature_columns, 
        'over_hit' # Dummy target, we won't use it
    )
    
    # Manually extract the regression target
    y_train = merged_train_data[target_col].values
    y_val = merged_test_data[target_col].values
    
    # Get the Vegas Line for the validation set (to calculate edge later)
    val_vegas_line = merged_test_data['total_line'].values

    # Convert features to Numpy
    X_train = X_train_t.numpy()
    X_val = X_val_t.numpy()

    print(f"Training shape: {X_train.shape}")

    # 5. Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50
    )

    # 6. Train
    print("\nTraining XGBoost Regressor...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # 7. Evaluate
    y_pred = model.predict(X_val)
    
    # Basic Metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"\nMean Absolute Error (MAE): {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
    
    # 8. THE BETTING SIMULATION (The Real Test)
    print("\n--- BETTING SIMULATION ---")
    
    # Calculate the 'Edge' (Model Prediction - Vegas Line)
    # Positive Edge = Model predicts MORE points -> Bet Over
    # Negative Edge = Model predicts FEWER points -> Bet Under
    edges = y_pred - val_vegas_line
    
    # Determine the bet and the result
    # Bet Over if Edge > 0, Bet Under if Edge < 0
    # Win if (Bet Over AND Actual > Line) OR (Bet Under AND Actual < Line)
    
    actual_diff = y_val - val_vegas_line
    
    # Create a DataFrame for analysis
    results_df = pd.DataFrame({
        'Prediction': y_pred,
        'Vegas': val_vegas_line,
        'Actual': y_val,
        'Edge': edges,
        'Actual_Diff': actual_diff
    })
    
    # Check win rates at different thresholds
    thresholds = [0, 1, 2, 3, 4, 5]
    
    for t in thresholds:
        # Filter games where our edge is greater than the threshold
        # We take absolute value because a -5 edge is just as strong as a +5 edge
        active_bets = results_df[abs(results_df['Edge']) >= t]
        
        if len(active_bets) == 0:
            print(f"Threshold {t}+: No bets found.")
            continue
            
        # Did we win?
        # Win Condition: (Edge > 0 and Actual > Vegas) OR (Edge < 0 and Actual < Vegas)
        wins = ((active_bets['Edge'] > 0) & (active_bets['Actual'] > active_bets['Vegas'])) | \
               ((active_bets['Edge'] < 0) & (active_bets['Actual'] < active_bets['Vegas']))
        
        # Pushes (Actual == Vegas) don't count as wins or losses usually, but let's count them as loss for strictness
        
        win_rate = wins.mean()
        print(f"Threshold {t}+ points: {len(active_bets)} bets | Win Rate: {win_rate*100:.2f}%")

    # 9. Feature Importance
    xgb.plot_importance(model, max_num_features=15)
    plt.title("Feature Importance (Regression)")
    plt.show()

if __name__ == "__main__":
    run_xgboost_regression()