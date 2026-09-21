import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MultimodalMamba(nn.Module):
    def __init__(self, video_dim=1280, audio_dim=25, d_model=192,
                 d_state=32, d_conv=4, expand=2, classifier_hidden=128,
                 num_classes=8, dropout=0.20):
        super().__init__()
        self.proj_v = nn.Sequential(nn.Linear(video_dim,d_model), nn.LayerNorm(d_model))
        self.proj_a = nn.Sequential(nn.Linear(audio_dim,d_model), nn.LayerNorm(d_model))
        self.mamba_v = Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand)
        self.mamba_a = Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand)
        fusion_dim=2*d_model
        self.mamba_fus=Mamba(d_model=fusion_dim,d_state=d_state,d_conv=d_conv,expand=expand)
        self.norm_fus=nn.LayerNorm(fusion_dim)
        self.classifier=nn.Sequential(nn.Linear(fusion_dim,classifier_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(classifier_hidden,num_classes))
    def forward(self,x_video,x_audio):
        v=self.mamba_v(self.proj_v(x_video)); a=self.mamba_a(self.proj_a(x_audio))
        x=self.norm_fus(self.mamba_fus(torch.cat([v,a],dim=-1)))
        return self.classifier(x.mean(dim=1))
