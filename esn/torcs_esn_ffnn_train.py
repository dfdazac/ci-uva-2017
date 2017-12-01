from simple_esn import SimpleESN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

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

X_test = inputs[split_idx:]
y_test = targets[split_idx:]

# Generate echoes from a reservoir
print("Generating echoes...")
n_readout = 500
my_esn = SimpleESN(n_readout=n_readout, n_components=n_readout,
                   damping=0.5, weight_scaling=1.0)
echo_train = my_esn.fit_transform(X_train)

# Train a feedforward neural network to learn
# the control actions given the echoes
model = SteeringClassifier(n_readout, 100, 7)
if CUDA:
    model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

EPOCHS = 20
print("Training neural network...")
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(len(echo_train)):
        # Forward propagate
        log_probs = model(get_variable(echo_train[i]))
        train_loss = loss_function(log_probs, get_variable([y_train[i]], dtype="long"))
        # Backward propagate
        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Log error
        total_loss += train_loss.data[0]
    print("{:d} - {:.2f}".format(epoch+1, total_loss))

# Evaluate accuracy
correct = 0
for n, data in enumerate(echo_train):
    v, i = model(autograd.Variable(torch.FloatTensor(data))).data.max(1)
    if i[0] == y_train[n]:
        correct += 1
print("Training accuracy: {:.1f}%".format(100 * correct/len(predictions)))
