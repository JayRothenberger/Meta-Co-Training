import torch
import torch.nn as nn
from copy import deepcopy as copy


class LinearProbe(torch.nn.Module):
    def __init__(self, module, size, num_classes, temperature=1.0):
        super().__init__()
        self.m = module

        self.m.train()

        self.linear = torch.nn.Linear(size, num_classes)
        self.temperature = temperature
    
    def forward(self, x):
        x = self.m(x).detach()
        return self.linear(x)  / self.temperature
    

class MLPProbe(torch.nn.Module):
    def __init__(self, module, size, num_classes, temperature=1.0):
        super().__init__()
        self.m = module

        self.m.train()
        # we call it this despite its nature to allow compatabilit with FPFT and FinetunedLinearProbe
        self.linear = FCNN(size, num_classes, [1024, 1024, 1024])
        self.temperature = temperature
    
    def forward(self, x):
        x = self.m(x).detach()
        return self.linear(x)  / self.temperature


class FPFT(torch.nn.Module):
    def __init__(self, module, temperature=1.0):
        super().__init__()
        self.m = copy(module.m)

        self.m.train()

        self.linear = copy(module.linear)
        self.temperature = temperature
    
    def forward(self, x):
        x = self.m(x)
        return self.linear(x) / self.temperature
    

class FinetunedLinearProbe(torch.nn.Module):
    def __init__(self, module, temperature=1.0):
        super().__init__()
        self.m = copy(module.m)

        self.m.train()

        self.linear = copy(module.linear)
        self.temperature = temperature
    
    def forward(self, x):
        x = self.m(x).detach()
        return self.linear(x) / self.temperature


class FCNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, normalize=nn.BatchNorm1d, skips=True, activation=nn.LeakyReLU(), dropout=0.0):
        """
        define the structure of the multilayer perceptron

        :int input_dim: number of input dimensions to the model

        :int output_dim: number of output dimensions of the model

        :callable normalize: normalization to apply after each activation

        :list hidden_dims: a list of hidden dimensions

        :bool skips: if True include skip connections, default True

        :callable activation: an activation function
        """
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        self.activation = activation
        self.normalize = normalize
        self.skips = skips
        skip = []
        
        if hidden_dims:
            if normalize is not None:
                self.layers.extend([nn.Linear(input_dim, hidden_dims[0]), torch.nn.Dropout(p=dropout, inplace=False), self.normalize(hidden_dims[0])])
            else:
                self.layers.extend([nn.Linear(input_dim, hidden_dims[0]), torch.nn.Dropout(p=dropout, inplace=False)])
                
            if self.skips:
                skip.append(hidden_dims[0])
                
            for i in range(len(hidden_dims[:-2])):
                if self.skips:
                    dim = hidden_dims[i + 1] + sum(skip)
                    skip.append(hidden_dims[i + 2])
                else:
                    dim = hidden_dims[i + 1]
                self.layers.extend([nn.Linear(dim, hidden_dims[i + 2]), torch.nn.Dropout(p=dropout, inplace=False)])
                if normalize is not None:
                    self.layers.extend([self.normalize(hidden_dims[i + 2])])
                    
            self.layers.extend([nn.Linear(hidden_dims[-1] + sum(skip), output_dim)])
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        

    def forward(self, x):
        skip = []
        for i, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                x = l(torch.concat(skip + [x], -1))
                x = self.activation(x)
            else:
                x = l(x)
            if (len(self.layers) > (i + 1)) and (self.normalize is None or i > 0):
                if self.skips and isinstance(self.layers[i+1], nn.Linear):
                    skip.append(x)
        return x