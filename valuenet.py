import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, Node_size):
        super().__init__()
        input_size = Node_size["input_size"]
        hidden_size = Node_size["hidden_size"]
        hidden_size2 = Node_size["hidden_size2"]
        output_size = Node_size["output_size"]


        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )
    def forward(self, x):
        return self.layers(x)