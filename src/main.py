import preprocess
import NN_dropout
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    merged_data = preprocess.merge_data()

    #string cols: roof, surface
    feature_columns = [
    #Market/context
    'total_line','over_odds','under_odds','spread_line','away_moneyline','home_moneyline',
    'away_spread_odds','home_spread_odds','week','temp','wind',
    'away_rest','home_rest', 'roof_closed','roof_dome','roof_open','roof_outdoors',
    'surface_a_turf','surface_astroturf','surface_fieldturf','surface_grass',
    'surface_grass','surface_matrixturf','surface_sportturf','surface_unknown',

    #Away team stats
    'away_passing_yards','away_passing_tds','away_passing_epa','away_rushing_yards','away_rushing_tds','away_rushing_epa',
    'away_receiving_yards','away_receiving_tds','away_receiving_epa','away_def_sacks','away_def_interceptions',
    #Home team stats
    'home_passing_yards','home_passing_tds','home_passing_epa','home_rushing_yards','home_rushing_tds','home_rushing_epa',
    'home_receiving_yards','home_receiving_tds','home_receiving_epa','home_def_sacks','home_def_interceptions',
]
    target_col = 'over_hit'

    X_train_t, X_val_t, y_train_t, y_val_t, scaler = preprocess.preprocess(merged_data, feature_columns, target_col)

    input_features = X_train_t.shape[1]

    DROPOUT_RATES = [0.1, 0.3, 0.5]
    LEARNING_RATE = 0.01
    NUM_ITERATIONS = 5000
    BATCH_SIZE = 5
    CHECK_EVERY = 500

    for dropout_rate in DROPOUT_RATES:
        print(f"\nTraining with Dropout Rate: {dropout_rate}")

        model = NN_dropout.FeedForwardNetWithDropout(input_size=input_features, dropout_rate=dropout_rate)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        train_losses, val_losses, train_accs, val_accs, iterations = NN_dropout.train_with_minibatch_dropout(
            model, criterion, optimizer,
            X_train_t, y_train_t,
            X_val_t, y_val_t,
            num_iterations=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            check_every=CHECK_EVERY
        )

        print(f"Final Training Accuracy: {train_accs[-1]*100:.2f}%")
        print(f"Final Validation Accuracy: {val_accs[-1]*100:.2f}%")




