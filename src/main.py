import preprocess
import NN_dropout
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    

    feature_columns2 = ['total_line','over_odds','under_odds','spread_line','away_moneyline','home_moneyline',
    'away_spread_odds','home_spread_odds','week','temp','wind','away_rest','home_rest',
    #Away team stats
    'away_passing_yards','away_passing_tds','away_passing_epa','away_rushing_yards','away_rushing_tds','away_rushing_epa',
    'away_receiving_yards','away_receiving_tds','away_receiving_epa','away_def_sacks','away_def_interceptions',
    #Home team stats
    'home_passing_yards','home_passing_tds','home_passing_epa','home_rushing_yards','home_rushing_tds','home_rushing_epa',
    'home_receiving_yards','home_receiving_tds','home_receiving_epa','home_def_sacks','home_def_interceptions',
    ]

    target_col = 'over_hit'

    X_train_t, X_val_t, y_train_t, y_val_t, scaler = preprocess.preprocess(merged_data, feature_columns2, target_col)

    input_features = X_train_t.shape[1]

    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.01
    NUM_ITERATIONS = 7500
    BATCH_SIZE = 16
    CHECK_EVERY = 250


    print(f"\nTraining with learning rate: {LEARNING_RATE}, iterations: {NUM_ITERATIONS}, dropout rate: {DROPOUT_RATE}\n")
    train_accs_final = []
    val_accs_final = []
    train_losses_all_runs = []
    val_losses_all_runs = [] #loss for coinflip is 0.693. Want around 0.67 or lower which corresponds to about 60% accuracy

    for _ in range(1):
        model = NN_dropout.FeedForwardNetWithDropout(input_size=input_features, dropout_rate=DROPOUT_RATE)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        saved_model, train_losses, val_losses, train_accs, val_accs, iterations, best_val_loss = NN_dropout.train_with_minibatch_dropout(
            model, criterion, optimizer,
            X_train_t, y_train_t,
            X_val_t, y_val_t,
            num_iterations=NUM_ITERATIONS,
            batch_size=BATCH_SIZE,
            check_every=CHECK_EVERY
        )

        train_accs_final.append(train_accs[-1])
        val_accs_final.append(val_accs[-1])
        train_losses_all_runs.append(train_losses)
        val_losses_all_runs.append(val_losses)

        #filename = f"Saved_models/model_{best_val_loss*100:.4f}.pth"
        #torch.save(saved_model.state_dict(), filename)


    avg_train_acc = np.mean(train_accs_final)
    std_train_acc = np.std(train_accs_final)
    avg_val_acc = np.mean(val_accs_final)
    std_val_acc = np.std(val_accs_final)
    print(f"Average Final Training Accuracy: {avg_train_acc*100:.2f}% ± {std_train_acc*100:.2f}%")
    print(f"Average Final Validation Accuracy: {avg_val_acc*100:.2f}% ± {std_val_acc*100:.2f}%")


    #Plotting
    train_losses = np.array(train_losses_all_runs)  # [n_runs, n_checkpoints]
    val_losses = np.array(val_losses_all_runs)
    avg_train_losses = np.mean(train_losses, axis=0)
    avg_val_losses = np.mean(val_losses, axis=0)
    iterations = np.arange(0, NUM_ITERATIONS+1, CHECK_EVERY)[:len(avg_train_losses)]  # or from your 'iterations' list

    plt.figure(figsize=(8,5))
    plt.plot(iterations, avg_train_losses, label='Train Loss')
    plt.plot(iterations, avg_val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




