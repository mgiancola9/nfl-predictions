import preprocess_chrono_weekly as preprocess
import NN_dropout
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import joblib


if __name__ == "__main__":

    #Set seeds for reproducibility
    #torch.manual_seed(42)
    #np.random.seed(42)

    merged_train_data = preprocess.merge_data("games_train.csv")
    merged_test_data = preprocess.merge_data("games_test.csv")

    feature_columns = ['total_line','over_odds','under_odds','spread_line','week','temp','wind','away_rest','home_rest',
                       
        #Away team stats
        'away_avg_passing_epa', 'away_avg_rushing_epa', 
        'away_avg_points_scored', 'away_avg_points_allowed', 'away_avg_total_yards_allowed',

        #Home team stats
        'home_avg_passing_epa', 'home_avg_rushing_epa', 
        'home_avg_points_scored', 'home_avg_points_allowed', 'home_avg_total_yards_allowed'
    ]

    target_col = 'over_hit'
    #print("Available Columns:", merged_train_data.columns.tolist())

    # Sanity Check: Print specific rows to manually verify shift
    # Check a random team's Week 1, 2, and 3
    #sample_team = merged_train_data[merged_train_data['home_team'] == 'BUF'].sort_values(['season', 'week'])
    #print(sample_team[['season', 'week', 'home_team', 'home_score', 'home_avg_points_scored']].head(5))
    X_train_t, X_val_t, y_train_t, y_val_t, scaler = preprocess.preprocess(merged_train_data, merged_test_data, feature_columns, target_col)

    input_features = X_train_t.shape[1]

    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    NUM_ITERATIONS = 7500
    BATCH_SIZE = 32
    CHECK_EVERY = 250
    MOMENTUM = 0
    WEIGHT_DECAY = 1e-3


    print(f"\nTraining with learning rate: {LEARNING_RATE}, iterations: {NUM_ITERATIONS}, dropout rate: {DROPOUT_RATE}\n")
    train_accs_final = []
    val_accs_final = []
    train_losses_all_runs = []
    val_losses_all_runs = [] #loss for coinflip is 0.693. Want around 0.67 or lower which corresponds to about 60% accuracy

    for _ in range(10):
        model = NN_dropout.FeedForwardNetWithDropout(input_size=input_features, dropout_rate=DROPOUT_RATE)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) #Using AdamW optimizer can choose others if wanted

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
        #joblib.dump(scaler, filename.replace('.pth', '_scaler.pkl'))


    #Final statistics
    avg_train_acc = np.mean(train_accs_final)
    std_train_acc = np.std(train_accs_final)
    avg_val_acc = np.mean(val_accs_final)
    std_val_acc = np.std(val_accs_final)
    print(f"Average Final Training Accuracy: {avg_train_acc*100:.2f}% ± {std_train_acc*100:.2f}%")
    print(f"Average Final Validation Accuracy: {avg_val_acc*100:.2f}% ± {std_val_acc*100:.2f}%")


    #Plotting
    train_losses = np.array(train_losses_all_runs)
    val_losses = np.array(val_losses_all_runs)
    avg_train_losses = np.mean(train_losses, axis=0)
    avg_val_losses = np.mean(val_losses, axis=0)
    iterations = np.arange(0, NUM_ITERATIONS+1, CHECK_EVERY)[:len(avg_train_losses)]

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




