import numpy as np
import matplotlib.pyplot as plt
from elm import ExtremeLearningMachine

# Import data
train_data = np.loadtxt("../data/train.csv", delimiter=",")
valid_data = np.loadtxt("../data/valid.csv", delimiter=",")
train_t = train_data[:, :3].T
train_x = train_data[:, 3:].T
valid_t = valid_data[:, :3].T
valid_x = valid_data[:, 3:].T

print(train_t.shape)

n_hidden_values = np.arange(10, 600, 50)
train_mse = np.zeros(len(n_hidden_values))
valid_mse = np.zeros(len(n_hidden_values))

# Each hidden unit value will be trained multiple times
n_trials = 10

for i, n_hidden in enumerate(n_hidden_values):
    print("Training with", n_hidden, "hidden units")
    best_mse = -1
    best_elm = None
    for j in range(n_trials):
        elm = ExtremeLearningMachine(train_x.shape[0], n_hidden, train_t.shape[0])
        elm.train(train_x, train_t)
        current_mse = elm.mse(train_x, train_t)
        if current_mse > best_mse:
            best_mse = current_mse
            best_elm = elm
            train_mse[i] = elm.mse(train_x, train_t)            

    valid_mse[i] = best_elm.mse(valid_x, valid_t)

plt.plot(n_hidden_values, train_mse, "-o", label="Training")
plt.plot(n_hidden_values, valid_mse, "-o", label="Validation")
plt.title("MSE")
plt.legend()
plt.xlabel("Epochs")
plt.show()