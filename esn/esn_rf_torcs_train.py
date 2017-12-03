import pickle
from simple_esn import SimpleESN
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

esn = pickle.load(open("models/reservoir.p", "rb"))

def train_random_forest(inputs, targets):

    # Split into training and validation sets
    split_idx = int(len(inputs) * 0.8)    
    x_train = inputs[:split_idx]
    y_train = targets[:split_idx]
    x_valid = inputs[split_idx:]
    y_valid = targets[split_idx:]

    # Train
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # Predict on training and validation sets
    y_train_pred = rf.predict(x_train)
    y_valid_pred = rf.predict(x_valid)

    # Report on f1 scores
    print("Training F1 score: {:.6f}".format(f1_score(y_train, y_train_pred)))
    print("Validation F1 score: {:.6f}".format(f1_score(y_valid, y_valid_pred)))

    return rf

ACCEL_COL = 0
BRAKE_COL = 1
filename = "../data/blackboard_quantized.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
inputs = esn.transform(data[:, 3:])
accel_targets = data[:, ACCEL_COL].astype(int)
brake_targets = data[:, BRAKE_COL].astype(int)

accel_model = train_random_forest(inputs, accel_targets)
brake_model = train_random_forest(inputs, brake_targets)

pickle.dump(accel_model, open("models/accel_model_v2.p", "wb"))
pickle.dump(brake_model, open("models/brake_model_v2.p", "wb"))

