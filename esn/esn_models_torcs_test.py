import pickle
from simple_esn import SimpleESN
import numpy as np
import torch
from ffnn_classifier import FFNNClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

esn = pickle.load(open("models/reservoir_v3.p", "rb"))

def evaluate_model(model, targets):
    predictions = np.zeros(len(targets))
    correct = 0
    for i in range(len(inputs)):    
        # Input to reservoir and then classify with model
        echo = esn.transform(inputs[i:i+1,:])
        predictions[i] = model.predict(echo)

        if predictions[i] == targets[i]:
            correct += 1

    accuracy = correct/len(targets)
    print("Accuracy: {:.6f}".format(accuracy))

    # Assume binary classes if max label is 1
    if max(targets) == 1:
        f1 = f1_score(targets, predictions)        
    else:
        f1 = f1_score(targets, predictions, average="weighted")
    print("F1 score: {:.6f}".format(f1))

    plt.plot(predictions, label="Predictions")
    plt.plot(targets, label="Targets")
    plt.legend()
    plt.show()

ACCEL_COL = 0
BRAKE_COL = 1
STEER_COL = 2
filename = "../data/blackboard_quantized.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
inputs = data[:, 3:]
accel_targets = data[:, ACCEL_COL].astype(int)
brake_targets = data[:, BRAKE_COL].astype(int)
steer_targets = data[:, STEER_COL].astype(int)

accel_model = pickle.load(open("models/accel_model_v3.p", "rb"))
brake_model = pickle.load(open("models/brake_model_v3.p", "rb"))
steer_model = FFNNClassifier(200, 250, 7)
steer_model.load_state_dict(torch.load("models/steer_model_v3.pt"))

accel_pred = evaluate_model(accel_model, accel_targets)
brake_pred = evaluate_model(brake_model, brake_targets)
steer_pred = evaluate_model(steer_model, steer_targets)


