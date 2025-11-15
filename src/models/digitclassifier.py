import torch.nn as nn


class DigitClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(  # Sequentially organizes layers in order of input
            nn.Linear(in_features, 512),
            nn.ReLU(),  # activation layer
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
