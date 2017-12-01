import torch.nn as nn
import torch.nn.functional as F

class MLPBinaryClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(MLPBinaryClassifier, self).__init__()
        
        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)
    
    def forward(self, inputs):
        h = F.tanh(self.linear1(inputs.view(1, -1)))
        y = self.linear2(h)
        return y
