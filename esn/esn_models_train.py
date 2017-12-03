import numpy as np
from simple_esn import SimpleESN
from sklearn.ensemble import RandomForestClassifier
from ffnn_classifier import train_ffnn_classifier
import pickle
import torch

# Load data
ACCEL_COL = 0
BRAKE_COL = 1
STEER_COL = 2
filename = "../data/blackboard_quantized.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
# Separate inputs (float) and targets (int)
sensors = data[:, 3:]
accel_targets = data[:, ACCEL_COL].astype(int)
brake_targets = data[:, BRAKE_COL].astype(int)
steer_targets = data[:, STEER_COL].astype(int)

n_samples = sensors.shape[0]
n_features = sensors.shape[1]

# Create the reservoir
esn = SimpleESN(n_features, n_readout=200,
    n_components=200, damping=0.99, weight_scaling=0.75)

# Generate echoes from the esn, which will be the input
# to the models
inputs = np.zeros((n_samples, esn.n_readout))
for i in range(n_samples):
    inputs[i, :] = esn.transform(sensors[i:i+1])

# Train acceleration
accel_model = RandomForestClassifier()
accel_model.fit(inputs, accel_targets)

# Train braking
brake_model = RandomForestClassifier()
brake_model.fit(inputs, brake_targets)

# Train steering
steer_model = train_ffnn_classifier(inputs, steer_targets, n_hidden=250,
    use_weights=True, verbose=True)

# Save results
pickle.dump(esn, open("models/reservoir_v3.p", "wb"))
pickle.dump(accel_model, open("models/accel_model_v3.p", "wb"))
pickle.dump(brake_model, open("models/brake_model_v3.p", "wb"))
torch.save(steer_model.state_dict(), "models/steer_model_v3.pt")

