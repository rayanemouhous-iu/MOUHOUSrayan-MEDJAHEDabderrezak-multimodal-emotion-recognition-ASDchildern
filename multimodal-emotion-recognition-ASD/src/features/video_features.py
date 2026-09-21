import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetFeatureExtractor:
    def __init__(self,device=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        weights=EfficientNet_B0_Weights.DEFAULT
        self.model=efficientnet_b0(weights=weights); self.model.classifier=nn.Identity(); self.model.eval().to(self.device)
        self.transform=weights.transforms()
    @torch.no_grad()
    def extract(self,frames):
        if isinstance(frames,list): frames=torch.stack([self.transform(f) for f in frames])
        return self.model(frames.to(self.device)).cpu().numpy()
