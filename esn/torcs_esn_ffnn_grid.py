from simple_esn import SimpleESN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

# Given the multiple set of hyperparameters that need to be
# tuned, this script performs a grid search on a feasible subset

CUDA = torch.cuda.is_available()

class SteeringClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(SteeringClassifier, self).__init__()
        
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, inputs):
        h = F.tanh(self.linear1(inputs.view(1, -1)))
        y = self.linear2(h)
        log_probs = F.log_softmax(y)
        return log_probs

def get_variable(x, dtype="float"):
    if dtype == "float":
        tensor = torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    elif dtype == "long":
        tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return autograd.Variable(tensor)


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

n_readouts = [200, 500, 1000]
dampings = [0.5, 0.75, 1.0]
scalings = [0.8, 0.9, 1.0]
hiddens = [[100, 200, 250], [400, 500, 700], [700, 1000, 1200]]

for n_r, n_readout in enumerate(n_readouts):
    for damping in dampings:
        for scaling in scalings:
            for n_hidden in hiddens[n_r]:
                # Generate echoes from a reservoir
                print("Readouts: {:d}  Damping: {:.2f}  Scaling: {:.2f}  Hiddens: {:d}".format(
                    n_readout, damping, scaling, n_hidden))

                my_esn = SimpleESN(n_readout=n_readout, n_components=n_readout,
                                   damping=damping, weight_scaling=scaling)
                echo_train = my_esn.fit_transform(X_train)
                valid_train = my_esn.transform(X_valid)

                # Train a feedforward neural network to learn
                # the control actions given the echoes
                model = SteeringClassifier(n_readout, n_hidden, 7)
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
                    else:
                        print("Terminating due to decrease in validation accuracy.")
                        break

                    print("{:^10.2f}{:^10.2f}".format(100*train_accuracy, 100*valid_accuracy))


