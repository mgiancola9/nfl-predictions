import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = DATA_DIR / "train_processed.csv"
TEST_PATH = DATA_DIR / "test_processed.csv"


# 1. Load data
def load_data():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Processed files not found. Expected:\n{TRAIN_PATH}\n{TEST_PATH}\n"
            "Run preprocess_nfl_data.py first."
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target_col = "fav_cover"  # our ATS target
    if target_col not in train.columns or target_col not in test.columns:
        raise ValueError(f"'{target_col}' not found in processed data. Check preprocessing.")

    # Drop target + any potential leak columns
    leak_cols = [
        "fav_cover",
        "home_cover",
        "home_margin",
        "ats_margin_home",
        "result",
    ]

    feat_train = train.drop(columns=[c for c in leak_cols if c in train.columns], errors="ignore")
    feat_test = test.drop(columns=[c for c in leak_cols if c in test.columns], errors="ignore")

    # Numeric-only features
    feat_train = feat_train.select_dtypes(include=[np.number])
    feat_test = feat_test.select_dtypes(include=[np.number])

    # Align columns between train and test
    common_cols = sorted(set(feat_train.columns) & set(feat_test.columns))

    dropped_train_only = sorted(set(feat_train.columns) - set(common_cols))
    dropped_test_only = sorted(set(feat_test.columns) - set(common_cols))

    if dropped_train_only:
        print(f"Dropping {len(dropped_train_only)} train-only cols (not in test).")
    if dropped_test_only:
        print(f"Dropping {len(dropped_test_only)} test-only cols (not in train).")

    X_train = feat_train[common_cols]
    X_test = feat_test[common_cols]

    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"Train positive rate (fav_cover=1): {y_train.mean():.3f}")
    print(f"Test  positive rate (fav_cover=1): {y_test.mean():.3f}")

    return X_train, y_train, X_test, y_test


# 2. Train model (with eval history)
def train_xgb_classifier(X_train, y_train, X_val, y_val):
    """
    Regularized baseline with evaluation history.
    Uses X_val/y_val as 'validation_1' for learning curve plotting.
    """
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=1.0,
        reg_lambda=5.0,
        reg_alpha=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    eval_set = [
        (X_train, y_train),  # 'validation_0'
        (X_val, y_val),      # 'validation_1'
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,  # no per-iteration spam
    )

    evals_result = model.evals_result()
    return model, evals_result


# 3. Plot learning curves
def plot_learning_curves(evals_result, output_path=None):
    """
    Plot train vs validation log loss over boosting rounds.

    Assumes keys:
      - 'validation_0': train
      - 'validation_1': validation (here: 2025 holdout)
    """
    if not evals_result or "validation_0" not in evals_result:
        print("No evals_result found; skipping learning curve plot.")
        return

    train_logloss = evals_result["validation_0"]["logloss"]

    # Look for any non-train validation key
    val_key = None
    for k in evals_result.keys():
        if k != "validation_0":
            val_key = k
            break

    val_logloss = evals_result[val_key]["logloss"] if val_key else None

    rounds = range(1, len(train_logloss) + 1)

    plt.figure()
    plt.plot(rounds, train_logloss, label="Train")
    if val_logloss is not None:
        plt.plot(rounds, val_logloss, label="Validation")
    plt.xlabel("Boosting Round")
    plt.ylabel("Log Loss")
    plt.title("XGBoost Learning Curve (Log Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Learning curve plot saved to: {output_path}")
    else:
        plt.show()


# 4. Evaluate
def evaluate_model(model, X_train, y_train, X_test, y_test):
    p_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = (p_train >= 0.5).astype(int)

    p_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (p_test >= 0.5).astype(int)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    brier = brier_score_loss(y_test, p_test)
    ll = log_loss(y_test, p_test)

    print("\n=== Model Performance (Favorite Cover) ===")
    print(f"Train Accuracy:         {acc_train:.3f}")
    print(f"Test  Accuracy (2025):  {acc_test:.3f}")
    print(f"Test  Brier Score:      {brier:.3f}")
    print(f"Test  Log Loss:         {ll:.3f}")

    # Naive baseline: always pick majority class ON TEST
    majority_class = 1 if y_test.mean() >= 0.5 else 0
    base_pred = np.full_like(y_test, majority_class)
    base_acc = accuracy_score(y_test, base_pred)
    print(f"\nNaive baseline accuracy (always {majority_class}): {base_acc:.3f}")

    edge = p_test - 0.5
    print(f"Avg |edge| on test:     {np.mean(np.abs(edge)):.3f}")

    return acc_test, base_acc


# 5. Main
def main():
    X_train, y_train, X_test, y_test = load_data()

    # Train with eval_set so we can plot learning curves.
    # Note: using 2025 as "validation" here is for diagnostics only.
    model, evals_result = train_xgb_classifier(X_train, y_train, X_test, y_test)

    # Evaluate as before
    acc_test, base_acc = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Plot learning curves
    plots_dir = BASE_DIR / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    curve_path = plots_dir / "learning_curve_logloss.png"
    plot_learning_curves(evals_result, output_path=str(curve_path))

    if acc_test > base_acc:
        print("\n Model beats naive baseline on 2025.")
    else:
        print("\n Model does NOT beat naive baseline yet. Focus next on feature engineering + market context.")


if __name__ == "__main__":
    main()
