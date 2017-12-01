import numpy as np
import matplotlib.pyplot as plt
from elm import ExtremeLearningMachine

np.random.seed(42)

def train(inputs, targets):
    split_idx = int(len(inputs) * 0.7)

    train_x = inputs[:split_idx].T
    train_t = targets[:split_idx].T

    valid_x = inputs[split_idx:].T
    valid_t = targets[split_idx:].T

    n_hidden_values = np.arange(10, 71, 10)
    train_mse = np.zeros(len(n_hidden_values))
    valid_mse = np.zeros(len(n_hidden_values))

    # Each hidden unit value will be trained multiple times
    n_trials = 50

    for i, n_hidden in enumerate(n_hidden_values):
        print("Training with", n_hidden, "hidden units")
        best_mse = float("inf")
        best_elm = None
        
        for j in range(n_trials):
            elm = ExtremeLearningMachine(train_x.shape[0], n_hidden, train_t.shape[0])
            elm.train(train_x, train_t)
            current_mse = elm.mse(train_x, train_t)
            if current_mse < best_mse:
                best_mse = current_mse
                best_elm = elm
                train_mse[i] = current_mse           

        valid_mse[i] = best_elm.mse(valid_x, valid_t)

    plt.plot(n_hidden_values, train_mse, "-o", label="Training")
    plt.plot(n_hidden_values, valid_mse, "-o", label="Validation")
    plt.title("MSE")
    plt.legend()
    plt.xlabel("Hidden layers")
    plt.show()

def attempt1000():
    data = np.loadtxt("../data/new_data.txt")
    np.random.shuffle(data)
    inputs = data[:, 0:1]
    targets = data[:, 1:2]
    train(inputs, targets)


attempt1000()