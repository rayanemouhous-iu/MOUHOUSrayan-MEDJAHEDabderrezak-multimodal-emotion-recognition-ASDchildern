import torch.nn as nn

class TransformerBranch(nn.Module):
    def __init__(self,input_dim,d_model=128,nhead=4,num_layers=1,dim_feedforward=256,dropout=.30):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(input_dim,d_model),nn.LayerNorm(d_model))
        layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers)
    def forward(self,x): return self.encoder(self.proj(x))

class MultimodalTransformer(nn.Module):
    def __init__(self,video_dim=1280,audio_dim=25,d_model=128,nhead=4,num_layers=1,dim_feedforward=256,dropout=.30,num_classes=8):
        super().__init__()
        self.video_branch=TransformerBranch(video_dim,d_model,nhead,num_layers,dim_feedforward,dropout)
        self.audio_branch=TransformerBranch(audio_dim,d_model,nhead,num_layers,dim_feedforward,dropout)
        f=2*d_model
        layer=nn.TransformerEncoderLayer(f,nhead,2*dim_feedforward,dropout,batch_first=True,activation='gelu')
        self.fusion=nn.TransformerEncoder(layer,num_layers)
        self.norm=nn.LayerNorm(f)
        self.classifier=nn.Sequential(nn.Linear(f,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
    def forward(self,x_video,x_audio):
        x=self.norm(self.fusion(__import__('torch').cat([self.video_branch(x_video),self.audio_branch(x_audio)],dim=-1)))
        return self.classifier(x.mean(dim=1))
