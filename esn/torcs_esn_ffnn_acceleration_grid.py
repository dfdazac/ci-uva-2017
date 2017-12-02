import numpy as np
import matplotlib.pyplot as plt
import pickle
from mlp_classifier import MLPClassifier
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

def get_variable(x, dtype="float"):
    if dtype == "float":
        tensor = torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    elif dtype == "long":
        tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return autograd.Variable(tensor)

CUDA = torch.cuda.is_available()#87.31 93.

esn = pickle.load(open("reservoir.p", "rb"))

# Load data and preprocess
data = np.loadtxt("../data/all_sensors_all_controls.csv", delimiter=",", skiprows=1)
accel_labeled = np.zeros(len(data), dtype=int)
levels = [0, 1]
delta = 0.1
for i, d in enumerate(data[:, 0]):
    for label, level in enumerate(levels):
        if abs(level - d) <= 0.1:
            accel_labeled[i] = label
            continue

# Separate inputs and targets
inputs = data[:, 3:]
targets = accel_labeled.tolist()
split_idx = int(0.8 * len(inputs))

# Separate training and validation sets
X_train = inputs[:split_idx]
y_train = targets[:split_idx]

X_valid = inputs[split_idx:]
y_valid = targets[split_idx:]

echo_train = esn.transform(X_train)
valid_train = esn.transform(X_valid)

n_hidden_values = [200, 250, 300, 400, 500]

for n_hidden in n_hidden_values:
    print("Training with", n_hidden, "hidden units")
    # Train a feedforward neural network to learn
    # the control actions given the echoes
    model = MLPClassifier(200, n_hidden, 2)
    if CUDA:
        model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    EPOCHS = 20
    print("{:^20s}".format("Accuracy (%)"))
    print("{:^10s}{:^10s}".format("Training", "Validation"))

    prev_valid_accuracy = 0
    for epoch in range(EPOCHS):    
        train_accuracy = 0
        valid_accuracy = 0

        for i in range(len(echo_train)):
            # Forward propagate
            log_probs = model(get_variable(echo_train[i]))
            train_loss = loss_function(log_probs, get_variable([y_train[i]], dtype="long"))
            # Backward propagate
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Accumulate training error
            prob, idx = log_probs.data.max(1)
            if idx[0] == y_train[i]:
                train_accuracy += 1

        # Training accuracy
        train_accuracy /= len(echo_train)

        # Validation accuracy
        for j in range(len(valid_train)):
            prob, idx = model(get_variable([valid_train[j]])).data.max(1)
            if idx[0] == y_valid[j]:
                valid_accuracy += 1
        valid_accuracy /= len(valid_train)

        if valid_accuracy >= prev_valid_accuracy:
            prev_valid_accuracy = valid_accuracy
            print("{:^10.2f}{:^10.2f}".format(100*train_accuracy, 100*valid_accuracy))
        else:
            print("Terminating due to decrease in validation accuracy:")
            print("{:^10.2f}{:^10.2f}".format(100*train_accuracy, 100*valid_accuracy))
            break

