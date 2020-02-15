import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = self.linear(x)
        return pred

torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))
print(model.bias)

# pylint: disable=not-callable
x = torch.tensor([1.])
print(model.forward(x))