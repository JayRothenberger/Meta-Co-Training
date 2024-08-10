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
        # we call it this despite its nature to allow compatability with FPFT and FinetunedLinearProbe
        self.linear = FCNN(size, num_classes, [1024, 1024, 1024])
        self.temperature = temperature
    
    def forward(self, x):
        with torch.no_grad():
            x = self.m(x).detach()
        return self.linear(x)  / self.temperature


class MLPProbe_ENS(torch.nn.Module):
    def __init__(self, module, size, num_classes, members=10):
        super().__init__()
        device = torch.cuda.current_device()
        self.m = module

        self.members = members

        self.linear_layers = torch.nn.ModuleList([FCNN(size, num_classes, [1024, 1024, 1024]).to(device) for _ in range(members)])

        self.params, self.buffers = torch.func.stack_module_state(self.linear_layers)

        for param in self.params:
            self.params[param] = torch.nn.Parameter(self.params[param])
            self.register_parameter(param.replace('.', '#'), self.params[param])

        self.base_model = [copy(self.linear_layers[0]).to('meta')]

    def linears_wrapper(self, params, buffers, data):
        return torch.func.functional_call(self.base_model[0], (params, buffers), (data,))

    def linears(self, data):
        return torch.vmap(self.linears_wrapper, randomness='same')(self.params, self.buffers, data.unsqueeze(0).expand(self.members, -1, -1))

    def update_covariance(self):
        pass
        
    def forward(self, x, with_variance=False):

        with torch.no_grad():
            x = self.m(x)

        predictions = self.linears(x)
        assert predictions.shape[0] == len(self.linear_layers)

        if self.train:
            return torch.sum(predictions, 0) / len(self.linear_layers)
        else:
            if with_variance:
                # return the prediction with the uncertainty value
                pred = (torch.sum(predictions, 0) / predictions.shape[0]) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
                pred_idx = torch.argmax(pred, -1)
                smax = torch.nn.functional.softmax(predictions, -1)
                unc = 1.0 - torch.gather(smax, -1, pred_idx.unsqueeze(-1)).squeeze()
                return pred, unc
            else:
                # return the calibrated prediction
                return (torch.sum(predictions, 0) / len(self.linear_layers)) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
        
        
    def to(self, device):
      super().to(device)

      # we overrode this method to send our parameters and buffers
      # it must be done with no grad to avoid non-leaf tensor error
      with torch.no_grad():
        for param in self.params:
            # this must be a parameter not the original or it will not require grad
            param_requires_grad = self.params[param].requires_grad
            self.params[param] = self.params[param].to(device)
            self.params[param].requires_grad = param_requires_grad

        for buffer in self.buffers:
            if self.buffers[buffer] is not None:
              buffer_requires_grad = self.buffers[buffer].requires_grad
              self.buffers[buffer] = self.buffers[buffer].to(device)
              self.buffers[buffer].requires_grad = buffer_requires_grad

      return self
    
    def parameters(self, recurse: bool = True):
        return [self.params[param] for param in self.params]
        # return super().parameters(recurse)
    def buffers(self, recurse: bool = True):
        return [self.buffers[buffer] for buffer in self.buffers]
        # return super().buffers(recurse)

    def state_dict(self, *args, **kwargs):
      state_dict1 = super().state_dict(*args, **kwargs)
      state_dict1.update({'params': copy(self.params), 'buffers': copy(self.buffers)})
      return state_dict1

    def load_state_dict(self, state_dict, *args, **kwargs):
        with torch.no_grad():
            for param in self.params:
                self.params[param].data = state_dict['params'][param].data

            self.buffers = state_dict['buffers']
            del state_dict['params']
            del state_dict['buffers']
        super().load_state_dict(state_dict, *args, **kwargs)


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