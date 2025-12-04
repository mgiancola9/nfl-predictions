#Actual Model file which will be deployed evantually
import xgboost as xgb
import preprocess_chrono_weekly as preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURATION ---
FEATURE_COLUMNS = [
    'total_line', 'spread_line', 'week', 'temp', 'wind', 'away_rest', 'home_rest',
    'away_avg_passing_epa', 'home_avg_passing_epa_allowed',
    'away_avg_rushing_epa', 'home_avg_rushing_epa_allowed',
    'home_avg_passing_epa', 'away_avg_passing_epa_allowed',
    'home_avg_rushing_epa', 'away_avg_rushing_epa_allowed',
    'away_avg_points_scored', 'home_avg_points_allowed',
    'home_avg_points_scored', 'away_avg_points_allowed'
]

def run_xgboost_regression():
    # 1. Load Data
    print("Loading data...")
    merged_train_data = preprocess.merge_data("games_train.csv")
    merged_test_data = preprocess.merge_data("games_test.csv")

    # 2. Create Regression Target
    print("Creating regression targets...")
    merged_train_data['actual_total'] = merged_train_data['home_score'] + merged_train_data['away_score']
    merged_test_data['actual_total'] = merged_test_data['home_score'] + merged_test_data['away_score']
    
    target_col = 'actual_total'

    # 3. Extract Features and Labels (No Scaling, No Tensors)
    print("Extracting features...")
    X_train, y_train = preprocess.get_features_and_labels(
        merged_train_data, FEATURE_COLUMNS, target_col
    )
    
    X_val, y_val = preprocess.get_features_and_labels(
        merged_test_data, FEATURE_COLUMNS, target_col
    )
    
    # Get the Vegas Line for the validation set (to calculate edge later)
    val_vegas_line = merged_test_data['total_line'].values

    print(f"Training shape: {X_train.shape}")

    # 4. Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000,
        learning_rate=0.005,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    # 5. Train
    print("\nTraining XGBoost Regressor...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # 6. Evaluate
    y_pred = model.predict(X_val)
    
    # Basic Metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"\nMean Absolute Error (MAE): {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
    
    # 7. THE BETTING SIMULATION
    print("\n--- BETTING SIMULATION ---")
    
    # Calculate Edge
    edges = y_pred - val_vegas_line
    
    # Create Analysis DataFrame
    results_df = pd.DataFrame({
        'Prediction': y_pred,
        'Vegas': val_vegas_line,
        'Actual': y_val,
        'Edge': edges
    })
    
    thresholds = [0, 1, 2, 3, 4, 5]
    
    for t in thresholds:
        active_bets = results_df[abs(results_df['Edge']) >= t]
        
        if len(active_bets) == 0:
            print(f"Threshold {t}+: No bets found.")
            continue
            
        wins = ((active_bets['Edge'] > 0) & (active_bets['Actual'] > active_bets['Vegas'])) | \
               ((active_bets['Edge'] < 0) & (active_bets['Actual'] < active_bets['Vegas']))
        
        win_rate = wins.mean()
        print(f"Threshold {t}+ points: {len(active_bets)} bets | Win Rate: {win_rate*100:.2f}%")

    # 8. Feature Importance
    xgb.plot_importance(model, max_num_features=15)
    plt.title("Feature Importance (Regression)")
    plt.show()

if __name__ == "__main__":
    run_xgboost_regression()