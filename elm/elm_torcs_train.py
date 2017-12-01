import numpy as np
import matplotlib.pyplot as plt
from elm import ExtremeLearningMachine
import pickle

np.random.seed(42)

def train(inputs, targets):
    split_idx = int(len(inputs) * 0.8)

    train_x = inputs[:split_idx].T
    train_t = targets[:split_idx].T

    valid_x = inputs[split_idx:].T
    valid_t = targets[split_idx:].T

    n_hidden = 20
    n_trials = 1000
    best_mse = float("inf")
    best_elm = None

    # Train
    for j in range(n_trials):
        print(j)
        elm = ExtremeLearningMachine(train_x.shape[0], n_hidden, train_t.shape[0])
        elm.train(train_x, train_t)
        
        current_mse = elm.mse(valid_x, valid_t)
        if current_mse < best_mse:
            best_mse = current_mse
            best_elm = elm

    print("Done training with validation MSE of {:.4f}".format(best_elm.mse(valid_x, valid_t)))

    # Export model
    pickle.dump(best_elm, open("elm_driver.p", "wb"))


def attempt1000():
    data = np.loadtxt("../data/new_data.txt")

    inputs = data[:, 0:1]
    targets = data[:, 1:2]
    train(inputs, targets)

attempt1000()
