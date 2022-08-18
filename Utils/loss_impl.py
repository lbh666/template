import torch.nn as nn

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, y, label):
        return self.loss(y, label)