import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, Node_size):
        super().__init__()
        input_size = Node_size["input_size"]
        hidden1 = Node_size["hidden1"]
        hidden2 = Node_size["hidden2"]
        output_size = Node_size["output_size"]


        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)
        )
    def forward(self, x):
        return self.layers(x)