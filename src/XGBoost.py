import xgboost as xgb
import preprocess_chrono_weekly as preprocess
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

def run_xgboost():
    merged_train_data = preprocess.merge_data("games_train.csv")
    merged_test_data = preprocess.merge_data("games_test.csv")

    feature_columns = ['total_line','spread_line','week','temp','wind','away_rest','home_rest',
                       
        #OFFENSE vs DEFENSE
        'away_avg_passing_epa', 'home_avg_passing_epa_allowed', # Away Pass vs Home Pass D
        'away_avg_rushing_epa', 'home_avg_rushing_epa_allowed', # Away Rush vs Home Rush D
        
        'home_avg_passing_epa', 'away_avg_passing_epa_allowed', # Home Pass vs Away Pass D
        'home_avg_rushing_epa', 'away_avg_rushing_epa_allowed', # Home Rush vs Away Rush D
        
        'away_avg_points_scored', 'home_avg_points_allowed',
        'home_avg_points_scored', 'away_avg_points_allowed'
    ]

    target_col = 'over_hit'

    X_train_t, X_val_t, y_train_t, y_val_t, scaler = preprocess.preprocess(
        merged_train_data, 
        merged_test_data, 
        feature_columns, 
        target_col
    )

    #Convert PyTorch Tensors to Numpy for XGBoost
    X_train = X_train_t.numpy()
    y_train = y_train_t.numpy().ravel() #Flatten to 1D array
    X_val = X_val_t.numpy()
    y_val = y_val_t.numpy().ravel()

    #print(f"Training shape: {X_train.shape}")
    #print(f"Validation shape: {X_val.shape}")

    #Initialize XGBoost
    #objective='binary:logistic' gives us probabilities
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,        # Maximum iterations (we'll stop early)
        learning_rate=0.01,       # Slower learning for better generalization
        max_depth=3,              # Keep trees shallow to prevent overfitting (critical for small data)
        subsample=0.8,            # Use 80% of data per tree
        colsample_bytree=0.8,     # Use 80% of features per tree
        random_state=42,
        eval_metric='logloss',     # Metric to watch for early stopping
        early_stopping_rounds=50 # Stop if validation loss doesn't improve for 50 rounds
    )

    #Train with Early Stopping
    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100               # Print progress every 100 rounds
    )

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    print(f"Final Training Accuracy:   {train_acc*100:.2f}%")

    #Get probability predictions (for log loss)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    #Get binary predictions (for accuracy)
    y_pred = model.predict(X_val)

    val_acc = accuracy_score(y_val, y_pred)
    val_loss = log_loss(y_val, y_pred_prob)

    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Final Validation Log Loss: {val_loss:.4f}")

    #Feature Importance
    #Which features actually matter.
    xgb.plot_importance(model, max_num_features=21)
    plt.title("Feature Importance")
    plt.show()

if __name__ == "__main__":
    run_xgboost()