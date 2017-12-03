import pickle
import numpy as np
from simple_esn import SimpleESN
import copy
from ffnn_classifier import train_ffnn_classifier
import torch

# Load data
STEER_COL = 2
filename = "../data/blackboard_quantized_norm.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)

# Separate inputs (float) and targets (int)
sensors = data[:, 3:]
targets = data[:, STEER_COL].astype(int).tolist()
n_samples = sensors.shape[0]
n_features = sensors.shape[1]
split_factor = 0.8
split_idx = int(len(targets) * split_factor)

# Hyperparameters
n_readouts = [300, 500]
dampings = [0.75, 0.9, 0.95]
hiddens = [[300, 350], [300, 350, 350]]

best_score = 0
best_param_str = ""
param_formatter = "readouts: {:d} damping: {:.2f} hiddens: {:d} score: {:.9f}"
val_targets = targets[split_idx:]
val_predictions = np.zeros(len(val_targets))

for n, n_readout in enumerate(n_readouts):
    for damping in dampings:
        for n_hidden in hiddens[n]:
            esn = SimpleESN(n_features, n_readout, n_readout, damping)
            # Generate echoes from the esn, which will be the input
            # to the FFNN
            inputs = np.zeros((n_samples, esn.n_readout))
            for i in range(n_samples):
                inputs[i, :] = esn.transform(sensors[i:i+1])

            # Train
            ffnn = train_ffnn_classifier(inputs, targets, n_hidden,
                split_factor=split_factor, use_weights=False, verbose=True)

            # Score ffnn
            val_inputs = inputs[split_idx:]
            correct = 0
            for i in range(len(val_inputs)):
                prediction = ffnn.predict(val_inputs[i:i+1])
                if prediction == val_targets[i]:
                    correct += 1
            score = correct/len(val_targets)

            param_str = param_formatter.format(n_readout, damping, n_hidden, score)
            print(param_str)

            if score > best_score:
                best_score = score
                best_ffnn = copy.deepcopy(ffnn)
                best_esn = copy.deepcopy(esn)
                best_param_str = param_str

print("Best parameters:")
print(best_param_str)
