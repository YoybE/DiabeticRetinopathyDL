import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = "EfficientNetB0Classifier"
        self.unet_output = None

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace classifier head for binary classification (Healthy vs Severe DR)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)