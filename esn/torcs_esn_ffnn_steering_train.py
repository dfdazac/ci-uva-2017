from simple_esn import SimpleESN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pickle
from mlp_classifier import MLPClassifier

CUDA = torch.cuda.is_available()

def get_variable(x, dtype="float"):
    if dtype == "float":
        tensor = torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    elif dtype == "long":
        tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return autograd.Variable(tensor)

def main():
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

    esn = SimpleESN(n_readout=200, n_components=200,
                       damping=1.0, weight_scaling=1.0)
    echo_train = esn.fit_transform(X_train)
    valid_train = esn.transform(X_valid)

    # Train a feedforward neural network to learn
    # the control actions given the echoes
    model = MLPClassifier(200, 250, 7)
    if CUDA:
        model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0009)

    EPOCHS = 20
    print("{:^20s}".format("Accuracy (%)"))
    print("{:^10s}{:^10s}".format("Training", "Validation"))

    prev_valid_accuracy = 0
    best_model = None
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
            best_model = model
        else:
            print("Terminating due to decrease in validation accuracy.")
            print("{:^10.2f}{:^10.2f}".format(100*train_accuracy, 100*valid_accuracy))
            break


    if best_model is not None:
        torch.save(best_model.state_dict(), "steer_model.pt")
        pickle.dump(esn, open("reservoir.p", "wb"))
        print("Saved best model.")

if __name__ == '__main__':
    main()
