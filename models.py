import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self._make_block(3, 16)
        self.conv2 = self._make_block(16, 32)
        self.conv3 = self._make_block(32, 64)
        self.conv4 = self._make_block(64, 128)
        self.conv5 = self._make_block(128, 128)

        self.flatten_layer = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*7*7, out_features=2048),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten_layer(x)
        # x = x.view(x.size(0), -1) # input, batch_size, C, 7, 7
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def _make_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

class AdvancedCNN(nn.Module):  # for Animals dataset
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(3, 16)
        self.conv2 = self.make_block(16, 32)
        self.conv3 = self.make_block(32, 64)
        self.conv4 = self.make_block(64, 128)
        self.conv5 = self.make_block(128, 128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    random_data = torch.randn(8, 3, 224, 224)
    model = CNN(num_classes=10)
    prediction = model(random_data)
    print(prediction.shape)
