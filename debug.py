import torch.nn as nn
import torch
loss = nn.CrossEntropyLoss()

input = torch.randn(1, 5)
target = torch.tensor(2)
output = loss(input, target)
print(output)