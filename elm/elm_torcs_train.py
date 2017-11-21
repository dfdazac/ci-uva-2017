import numpy as np
import matplotlib.pyplot as plt
from elm import ExtremeLearningMachine
import pickle

# Import data
train_data = np.loadtxt("../data/train.csv", delimiter=",")
valid_data = np.loadtxt("../data/valid.csv", delimiter=",")
train_t = train_data[:, :3].T
train_x = train_data[:, 3:].T
valid_t = valid_data[:, :3].T
valid_x = valid_data[:, 3:].T

n_hidden = 500
n_trials = 20
best_mse = -1
best_elm = None

# Train
for j in range(n_trials):
    print(j)
    elm = ExtremeLearningMachine(train_x.shape[0], n_hidden, train_t.shape[0])
    elm.train(train_x, train_t)
    
    current_mse = elm.mse(valid_x, valid_t)
    if current_mse > best_mse:
        best_mse = current_mse
        best_elm = elm

print("Done training with validation MSE of {:.2f}".format(elm.mse(valid_x, valid_t)))

# Export model
pickle.dump(best_elm, open("elm_driver.p", "wb"))