import torch
import torch.nn as nn

class FeedForwardNetWithDropout(nn.Module):

    def __init__(self, input_size, dropout_rate): #dropout_rate between 0 and 1 (0.1, 0.3, 0.5)
        super(FeedForwardNetWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, 64) #first hidden layer with 64 neurons
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32) #second hidden layer with 32 neurons
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return self.sigmoid(x)
    

def calculate_accuracy(model, X, y):
    model.eval()  #Same accuracy calculation as before

    with torch.no_grad():
        outputs = model(X)                                   #Model predictions (probabilities)
        predictions = (outputs >= 0.5).float()               #Threshold at 0.5 for binary classification
        correct = (predictions == y).float().sum().item()    #Count correct predictions
        accuracy = correct / y.size(0)                       #Divide by total number of samples

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

    dataset_size = X_train.size(0)

    for i in range(1, num_iterations + 1):
        #Sample a random batch
        idx = torch.randint(0, dataset_size, (batch_size,))
        x_batch = X_train[idx]
        y_batch = y_train[idx]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        #Record losses and accuracies at intervals
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

    return train_losses, val_losses, train_accs, val_accs, iterations



