#Copy of XGBoost_Regression.py with added plots and CSV for report
import xgboost as xgb
import preprocess_chrono_weekly as preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def run_analysis():
    # 1. Load Data
    print("Loading data...")
    merged_train_data = preprocess.merge_data("games_train.csv")
    merged_test_data = preprocess.merge_data("games_test.csv")

    # 2. Create Regression Target
    merged_train_data['actual_total'] = merged_train_data['home_score'] + merged_train_data['away_score']
    merged_test_data['actual_total'] = merged_test_data['home_score'] + merged_test_data['away_score']
    
    target_col = 'actual_total'

    # 3. Extract Features (Keeping DataFrames for metadata later)
    X_train = merged_train_data[FEATURE_COLUMNS]
    y_train = merged_train_data[target_col]
    
    X_val = merged_test_data[FEATURE_COLUMNS]
    y_val = merged_test_data[target_col]

    print(f"Training shape: {X_train.shape}")

    # 4. Initialize XGBoost Regressor
    # Note: Reduced n_estimators slightly to prevent total overfitting if learning rate is low
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

    # 5. Train with Evaluation Logging
    print("\nTraining XGBoost Regressor...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # 6. Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"\nMAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # --- REPORT ARTIFACTS ---
    print("\nGenerating Report Artifacts...")

    # Plot 1: Learning Curves (Training vs Validation Loss)
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
    plt.plot(x_axis, results['validation_1']['rmse'], label='Validation')
    plt.legend()
    plt.ylabel('RMSE (Points)')
    plt.xlabel('Iterations')
    plt.title('XGBoost Learning Curve: Training vs Validation Loss')
    plt.grid(True)
    plt.savefig('report_learning_curve.png')
    print("- Saved report_learning_curve.png")

    # Plot 2: Predicted vs Actual Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, y_pred, alpha=0.5, color='blue')
    
    # Ideal line (Perfect prediction)
    max_val = max(max(y_val), max(y_pred))
    min_val = min(min(y_val), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Total Score')
    plt.ylabel('Predicted Total Score')
    plt.title('Predicted vs. Actual Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('report_scatter_pred_vs_actual.png')
    print("- Saved report_scatter_pred_vs_actual.png")

    # Plot 3: Residual Histogram (Error Distribution)
    residuals = y_val - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.savefig('report_residual_hist.png')
    print("- Saved report_residual_hist.png")

    # 7. EXPORT CSV (The "Audit Trail")
    # We reconstruct the dataframe using the index from merged_test_data
    
    # Get Vegas Lines from original data
    vegas_lines = merged_test_data['total_line']
    
    # Calculate Bet Recommendations
    edges = y_pred - vegas_lines
    recommendations = []
    
    # You can tune this "no bet" zone. Currently set to strict > 0.
    for edge in edges:
        if edge > 0:
            recommendations.append("OVER")
        elif edge < 0:
            recommendations.append("UNDER")
        else:
            recommendations.append("PASS") # Exact push

    # Determine Win/Loss
    # Win = (Rec=OVER & Actual>Vegas) OR (Rec=UNDER & Actual<Vegas)
    results = []
    for rec, actual, vegas in zip(recommendations, y_val, vegas_lines):
        if rec == "OVER":
            if actual > vegas: results.append("WIN")
            elif actual < vegas: results.append("LOSS")
            else: results.append("PUSH")
        elif rec == "UNDER":
            if actual < vegas: results.append("WIN")
            elif actual > vegas: results.append("LOSS")
            else: results.append("PUSH")
        else:
            results.append("NO BET")

    export_df = pd.DataFrame({
        'Season': merged_test_data['season'],
        'Week': merged_test_data['week'],
        'Home': merged_test_data['home_team'],
        'Away': merged_test_data['away_team'],
        'Vegas_Line': vegas_lines,
        'Predicted': np.round(y_pred, 1),
        'Actual': y_val,
        'Edge': np.round(edges, 2),
        'Recommendation': recommendations,
        'Result': results
    })
    
    export_df.to_csv('simulation_results.csv', index=False)
    print("- Saved simulation_results.csv")
    
    # Quick Summary for Console
    win_rate = len(export_df[export_df['Result']=='WIN']) / len(export_df[export_df['Result'].isin(['WIN', 'LOSS'])])
    print(f"\nOverall Win Rate: {win_rate*100:.2f}%")

if __name__ == "__main__":
    run_analysis()