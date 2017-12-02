from collections import Counter
from ffnn_classifier import FFNNClassifier, get_tensor, get_variable
import torch
import torch.nn as nn
import torch.optim as optim
import copy

CUDA = torch.cuda.is_available()

def train_esn_ffnn(inputs, targets, n_hidden, esn, split_factor=0.8,
    use_weights=False, verbose=False):
    """ Trains a classifier on temporal data, consisting of
    an echo state network and a feedforward neural network.
    Args:
        - inputs (N by n float array): historical features, where
            N is the number of samples and n is the dimension
            of the input.
        - targets (list): the corresponding class label (int)
            for each sample in the inverval [0, k-1] where
            k is the number of classes.
        - split_factor (float): a value in the interval (0, 1)
            that determines the fraction of the data used for
            training. The rest is used for validation.
        - use_weights (bool): whether to use weights or not,
            recommended if the classes are unbalanced.
        - verbose (bool): if True, progress is printed during training.

    Returns:
        - MLPClassifier, the best classifier based on the
            lowest loss in a validation set.
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

    # Get the echoes from the esn
    echo_train = esn.transform(x_train)
    echo_valid = esn.transform(x_valid)

    # Setup the model
    model = FFNNClassifier(esn.n_readout, n_hidden, n_labels)
    if CUDA:
        model.cuda()
    loss_function = nn.NLLLoss(class_weights, size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0009)

    # Start training
    EPOCHS = 20
    prev_loss = float("inf")
    best_model = None
    for epoch in range(EPOCHS):
        total_train_loss = 0
        
        for i in range(len(echo_train)):
            # Forward propagate
            log_probs = model(get_variable(echo_train[i:i+1]))
            train_loss = loss_function(log_probs, get_variable([int(y_train[i])], dtype="long"))
            # Backward propagate
            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.data[0]

        # Evaluate on validation set
        valid_log_probs = model(get_variable(echo_valid))
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







