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
split_idx = int(0.8 * len(inputs))

# Separate training and validation sets
X_train = inputs[:split_idx]
y_train = targets[:split_idx]

X_valid = inputs[split_idx:]
y_valid = targets[split_idx:]

echo_train = my_esn.transform(X_train)
valid_train = my_esn.transform(X_valid)

# Validation accuracy
valid_accuracy = 0
predictions = []
for j in range(len(echo_train)):
    prob, idx = model(get_variable([echo_train[j]])).data.max(1)
    if idx[0] == y_train[j]:
        valid_accuracy += 1
    predictions.append(idx[0])

valid_accuracy /= len(echo_train)
print(100 * valid_accuracy)

plt.plot(y_train, label="Target")
plt.plot(predictions, label="Prediction")
plt.legend()
plt.show()