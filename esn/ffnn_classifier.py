from collections import Counter
import copy
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

CUDA = torch.cuda.is_available()

def get_tensor(x, dtype="float"):
    if dtype == "float":
        return torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    if dtype == "long":
        return torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)

def get_variable(x, dtype="float"):
    return autograd.Variable(get_tensor(x, dtype))

class FFNNClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(FFNNClassifier, self).__init__()
        
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, inputs):
        h = F.tanh(self.linear1(inputs.view(len(inputs), -1)))
        y = self.linear2(h)
        log_probs = F.log_softmax(y)
        return log_probs

    def predict(self, x):
        log_probs = self.forward(get_variable(x))
        _, idx = log_probs.data.max(1)
        return idx[0]

def train_ffnn_classifier(inputs, targets, n_hidden, split_factor=0.8,
    use_weights=False, verbose=False):
    """ Trains an FFNNClassifier.
    Args:
        - inputs (N by n float array): historical features, where
            N is the number of samples and n is the dimension
            of the input.
        - targets (list): the corresponding class label (int)
            for each sample in the inverval [0, k-1] where
            k is the number of classes.
        - n_hidden (int): number of units in the hidden layer.
        - split_factor (float): a value in the interval (0, 1)
            that determines the fraction of the data used for
            training. The rest is used for validation.
        - use_weights (bool): whether to use weights or not,
            recommended if the classes are unbalanced.
        - verbose (bool): if True, progress is printed during training.
    Returns:
        - FFNNClassifier, the best classifier based on the
            lowest loss in the validation set.
    """
    # Split into training and validation sets
    split_idx = int(len(targets) * split_factor)

    x_train = inputs[:split_idx]
    y_train = targets[:split_idx]

    x_valid = inputs[split_idx:]
    y_valid = targets[split_idx:]

    # Here we assume that the training set contains
    # at least one sample of all labels        
    class_counts = Counter(y_train)
    n_labels = len(class_counts)

    # Determine class weights if needed
    class_weights = None
    if use_weights:        
        weights = [1/class_counts[i] for i in range(len(class_counts))]
        class_weights = get_tensor(weights)

    # Setup the model
    model = FFNNClassifier(inputs.shape[1], n_hidden, n_labels)
    if CUDA:
        model.cuda()
    loss_function = nn.NLLLoss(class_weights, size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Start training
    EPOCHS = 20
    prev_loss = float("inf")
    best_model = None
    for epoch in range(EPOCHS):
        total_train_loss = 0
        
        for i in range(len(x_train)):
            # Forward propagate
            log_probs = model(get_variable(x_train[i:i+1]))
            train_loss = loss_function(log_probs, get_variable([int(y_train[i])], dtype="long"))
            # Backward propagate
            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.data[0]

        # Evaluate on validation set
        valid_log_probs = model(get_variable(x_valid))
        total_valid_loss = loss_function(valid_log_probs, get_variable(y_valid, dtype="long")).data[0]
        
        # If loss improves, store model and continue
        if total_valid_loss <= prev_loss:
            prev_loss = total_valid_loss
            best_model = copy.deepcopy(model)
            if verbose:
                print("{:d}/{:d}: {:.6f}  {:.6f}".format(epoch + 1, EPOCHS, total_train_loss, total_valid_loss))
        # Otherwise terminate early
        else:        
            break

    return best_model


