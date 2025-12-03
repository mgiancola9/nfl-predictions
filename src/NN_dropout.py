import torch
import torch.nn as nn
import copy

class FeedForwardNetWithDropout(nn.Module):

    def __init__(self, input_size, dropout_rate): #dropout_rate between 0 and 1
        super(FeedForwardNetWithDropout, self).__init__()
        
        self.main = nn.Sequential(
                    # Layer 1: Compress 60 features to 32
                    nn.Linear(input_size, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    # Layer 2: Refine to 16
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    # Output Layer
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

    def forward(self, x):
        return self.main(x)

def calculate_accuracy(model, X, y):
    model.eval()

    with torch.no_grad():
        outputs = model(X)                                   #Model predictions (probabilities)
        predictions = (outputs >= 0.5).float()               #Threshold at 0.5 for binary classification
        correct = (predictions == y).float().sum().item()    #Count correct predictions
        accuracy = correct / y.size(0)                       #Divide by total number of samples

    model.train()
    return accuracy

def calculate_full_loss(model, criterion, X, y):
    """Helper function to calculate loss over an entire dataset."""
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs = model(X)
        loss = criterion(outputs, y)
    model.train() # Set model back to train mode
    return loss.item()

def train_with_minibatch_dropout(model, criterion, optimizer, X_train, y_train, X_val, y_val,
                                 num_iterations, batch_size, check_every):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    iterations = []

    # Track the best model and val loss
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_iteration = 0

    dataset_size = X_train.size(0)

    for i in range(1, num_iterations + 1):
        # Sample a random batch
        idx = torch.randint(0, dataset_size, (batch_size,))
        x_batch = X_train[idx]
        y_batch = y_train[idx]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # Record losses and accuracies at intervals
        if i % check_every == 0 or i == 1:
            train_loss = calculate_full_loss(model, criterion, X_train, y_train)
            val_loss = calculate_full_loss(model, criterion, X_val, y_val)
            train_acc = calculate_accuracy(model, X_train, y_train)
            val_acc = calculate_accuracy(model, X_val, y_val)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            iterations.append(i)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                early_stop_iteration = i

            model.train() #ensure model back in train mode
    
    #At end, restore and return best model and its metrics
    model.load_state_dict(best_model_state)
    print(f"Early stopping at iteration {early_stop_iteration} with best val loss {best_val_loss:.4f}")
    best_val_acc = calculate_accuracy(model, X_val, y_val)
    print(f"Validation Accuracy at best model: {best_val_acc*100:.2f}%")


    return model, train_losses, val_losses, train_accs, val_accs, iterations, best_val_loss