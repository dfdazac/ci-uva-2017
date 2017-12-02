import pickle
from simple_esn import SimpleESN
from esn_ffnn_train import train_esn_ffnn, get_variable
import numpy as np
import torch
from sklearn.metrics import f1_score
import copy

def kfold_indices(N, k):
    all_indices = np.arange(N,dtype=int)
    idx = [int(i) for i in np.floor(np.linspace(0,N,k+1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold+1]]
        valid_folds.append(valid_indices)
        train_folds.append(np.setdiff1d(all_indices, valid_indices))
    return train_folds, valid_folds

# Load data
STEER_COL = 2
filename = "../data/blackboard_quantized.csv"
esn = pickle.load(open("models/reservoir.p", "rb"))
data = np.loadtxt(filename, delimiter=",", skiprows=1)

# Separate inputs (float) and targets (int)
inputs = data[:, 3:]
targets = data[:, STEER_COL].astype(int)
folds = 4
train_folds, valid_folds = kfold_indices(len(targets), folds)

# Hyperparameters set
hiddens = [100, 150, 200, 250, 300]

best_score = 0
best_hidden = -1

for n_hidden in hiddens:
    fold_score = 0
    for k in range(folds):
        # Get training folds
        x_train_fold = inputs[train_folds[k]]
        y_train_fold = targets[train_folds[k]]

        # Train a new model
        model = train_esn_ffnn(x_train_fold, y_train_fold, n_hidden, esn,
            use_weights=True, verbose=True)

        # Evaluate on validation folds
        x_valid_fold = inputs[valid_folds[k]]
        y_valid_fold = targets[valid_folds[k]]
        y_valid_pred = np.zeros(len(y_valid_fold))        
        for i in range(len(x_valid_fold)):
            # Get echo for current sample and classify
            echo = esn.transform(x_valid_fold[i:i+1,:])
            y_valid_pred[i] = model.predict(echo)
        
        # Score based on validation predictions
        fold_score += f1_score(y_valid_fold, y_valid_pred, average="micro")
    # Average fold scores
    fold_score /= folds
    print("n_hidden: {:d}  f1: {:.6f}".format(n_hidden, fold_score))
    # Update best model
    if fold_score > best_score:
        best_score = fold_score
        best_hidden = n_hidden

with open("esn_ffnn_torcs_cv_results.log") as file:
    file.write("Best score obtained with {:d} hidden units.\n".format(best_hidden))









