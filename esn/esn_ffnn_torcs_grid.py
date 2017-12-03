import pickle
import numpy as np
from simple_esn import SimpleESN
import copy
from ffnn_classifier import train_ffnn_classifier
import torch

# Load data
STEER_COL = 2
filename = "../data/blackboard_quantized.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)

# Separate inputs (float) and targets (int)
sensors = data[:, 3:]
targets = data[:, STEER_COL].astype(int).tolist()
n_samples = sensors.shape[0]
n_features = sensors.shape[1]
split_factor = 0.8
split_idx = int(len(targets) * split_factor)

# Hyperparameters
n_readouts = [200, 300, 500]
dampings = [0.5, 0.75, 1.0]
scalings = [0.75, 1.0, 1.25]
hiddens = [[200, 250], [250, 300, 350], [400, 500, 600]]

best_score = 0
val_targets = targets[split_idx:]
val_predictions = np.zeros(len(val_targets))

for n, n_readout in enumerate(n_readouts):
    for damping in dampings:
        for scaling in scalings:
            for n_hidden in hiddens[n]:
                esn = SimpleESN(n_features, n_readout, n_readout, damping, scaling)
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
                score = correct/len(targets)

                print("readouts: {:d} damping: {:.2f} scaling: {:.2f} hiddens: {:d} score: {:.6f}".format(
                    n_readout, damping, scaling, n_hidden, score))

                if score > best_score:
                    best_score = score
                    best_ffnn = copy.deepcopy(ffnn)
                    best_esn = copy.deepcopy(esn)

torch.save(best_ffnn.state_dict(), "models/steer_model_v4.pt")
pickle.dump(best_esn, open("models/reservoir_v4.pt", "wb"))
