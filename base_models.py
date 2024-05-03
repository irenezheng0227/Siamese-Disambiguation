import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Fixed dropout rate for all layers
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Weight initialization
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        output = self.model(x)
        normalized_output = F.normalize(output, p=2, dim=1)  # Changed normalization to L2 norm
        return normalized_output


import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_2(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(MLP_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            #nn.ReLU(),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_1, hidden_size_2),
            #nn.ReLU(),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_2, output_size),
        )

        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[4].weight)
        nn.init.xavier_uniform_(self.model[8].weight)

    def forward(self, x):
        output = self.model(x)
        normalized_output = F.normalize(output, p=2, dim=1)  # Normalize along dimension 1 (the output vector dimension)
        return output#normalized_output

class MLP_3(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size):
        super(MLP_3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.BatchNorm1d(hidden_size_3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_3, output_size),
        )

        #Weight initialization
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[4].weight)
        nn.init.xavier_uniform_(self.model[8].weight)
        nn.init.xavier_uniform_(self.model[12].weight)
        
    def forward(self, x):
        output = self.model(x)
        normalized_output = F.normalize(output, p=1, dim=1)
        return normalized_output #self.model(x)
