import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self, num_classes= 10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3 * 32 * 32, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

if __name__ == '__main__':
    model = MyNeuralNetwork()
    random_input = torch.randn(8, 3, 32, 32)
    result = model(random_input)
    print(result.shape)
