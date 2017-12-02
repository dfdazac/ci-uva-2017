import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

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


