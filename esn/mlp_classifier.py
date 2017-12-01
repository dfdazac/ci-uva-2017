import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(MLPClassifier, self).__init__()
        
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, inputs):
        h = F.tanh(self.linear1(inputs.view(1, -1)))
        y = self.linear2(h)
        log_probs = F.log_softmax(y)
        return log_probs
