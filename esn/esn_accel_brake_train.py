import numpy as np
from simple_esn import SimpleESN
from sklearn.ensemble import RandomForestClassifier
from ffnn_classifier import train_ffnn_classifier
import pickle
import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

def evaluate_model(model, inputs, targets):
    # predictions = np.zeros(len(targets))
    # correct = 0
    # for i in range(len(inputs)):
    #     predictions[i] = model.predict(inputs[i:i+1])

    #     if predictions[i] == targets[i]:
    #         correct += 1

    # accuracy = correct/len(targets)
    #print("Accuracy: {:.6f}".format(accuracy))
    predictions = model.predict(inputs)

    # Assume binary classes if max label is 1
    if max(targets) == 1:
        f1 = f1_score(targets, predictions)        
    else:
        f1 = f1_score(targets, predictions, average="weighted")
    print("F1 score: {:.6f}".format(f1))

    plt.plot(targets, "o", lw=0.5, label="Targets")
    plt.plot(predictions, "x", lw=0.5, label="Predictions")    
    plt.legend()
    plt.show()

# Load data
ACCEL_COL = 0
BRAKE_COL = 1
filename = "../data/blackboard_quantized.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
# Separate inputs (float) and targets (int)
sensors = data[:, 3:]
accel_targets = data[:, ACCEL_COL].astype(int)
brake_targets = data[:, BRAKE_COL].astype(int)
split_idx = int(len(sensors) * 0.3)

n_samples = sensors.shape[0]
n_features = sensors.shape[1]

# Get the reservoir
esn = pickle.load(open("models/reservoir_v3.p", "rb"))

# Generate echoes from the esn, which will be the input
# to the models
inputs = np.zeros((n_samples, esn.n_readout))
for i in range(n_samples):
    inputs[i, :] = esn.transform(sensors[i:i+1])

# Train acceleration
#accel_model = RandomForestClassifier()
#accel_model = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
#accel_model = AdaBoostClassifier()
#accel_model.fit(inputs[split_idx:], accel_targets[split_idx:])

# Train braking
#brake_model = RandomForestClassifier()
#brake_model = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
#brake_model = AdaBoostClassifier()
brake_model.fit(inputs, brake_targets)

#accel_pred = evaluate_model(accel_model, inputs[:split_idx], accel_targets[:split_idx])
brake_pred = evaluate_model(brake_model, inputs[:split_idx], brake_targets[:split_idx])

pickle.dump(brake_model, open("models/brake_model_v4.p", "wb"))
