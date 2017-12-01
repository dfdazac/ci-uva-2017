from torcs_esn_ffnn_train import SteeringClassifier, get_variable
import torch
import pickle
from simple_esn import SimpleESN
import numpy as np
import matplotlib.pyplot as plt

# Load models
model = SteeringClassifier(200, 100, 7)
model.load_state_dict(torch.load("model.pt"))
my_esn = pickle.load(open("reservoir.p", "rb"))

# Load data and preprocess
data = np.loadtxt("../data/all_sensors_all_controls.csv", delimiter=",", skiprows=1)
steer_labeled = np.zeros(len(data), dtype=int)
levels = [-1, -3/4, -1/3, 0, 1/3, 3/4, 1]
delta = 0.1
for i, d in enumerate(data[:, 2]):
    for label, level in enumerate(levels):
        if abs(level - d) <= 0.1:
            steer_labeled[i] = label
            continue

# Separate inputs and targets
inputs = data[:, 3:]
targets = steer_labeled.tolist()

predictions = []

correct = 0
for i in range(len(inputs)):    
    # Input to reservoir
    echo = my_esn.transform(inputs[i:i+1,:])
    # Readout and input to NN
    prob, idx = model(get_variable(echo)).data.max(1)
    predictions.append(idx[0])
    if idx[0] == targets[i]:
        correct += 1

print(correct/len(targets) * 100)

plt.plot(predictions, label="Predictions")
plt.plot(targets, label="Targets")
plt.legend()
plt.show()

print(inputs.shape)