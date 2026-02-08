import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from valuenet import MLP
# import sqlite3

# conn = sqlite3.connect(":memory:")
# df = pd.read_sql(file)

if __name__ == "__main__":
    torch.manual_seed(42)
    X,y = X, y


input_size = 34
hidden1 = 64
hidden2 = 32
output_size = 1

node = {"input_size": input_size, 
        "hidden1": hidden1, 
        "hidden2": hidden2, 
        "output_size": output_size}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(node)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_num = 10
num_batch = 0

trainloader = torch.utils.data.DataLoader(dataset, batch_size= 10, num_workers= 1)

for epoch in range(epoch_num):
    print(f'Starting Epoch {epoch+1}')
    
    current_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float() 
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()

        outputs = MLP(inputs)

        loss = loss_fn(outputs, targets)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if i%10 == 0:
            print(f'Loss after mini-batch %5d: %.3f'%(i+1, current_loss/500))
            current_loss = 0.0

    print(f'Epoch {epoch+1} finished')

print("Training has completed")

.eval()