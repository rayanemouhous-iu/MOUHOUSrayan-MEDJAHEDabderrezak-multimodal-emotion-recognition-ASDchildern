import torch
import torch.nn as nn

class BiLSTMBranch(nn.Module):
    def __init__(self,input_dim,proj_dim=128,hidden=64,dropout=.30):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(input_dim,proj_dim),nn.LayerNorm(proj_dim)); self.lstm=nn.LSTM(proj_dim,hidden,batch_first=True,bidirectional=True); self.drop=nn.Dropout(dropout)
    def forward(self,x): return self.drop(self.lstm(self.proj(x))[0])

class MultimodalBiLSTM(nn.Module):
    def __init__(self,video_dim=1280,audio_dim=25,proj_dim=128,branch_hidden=64,fusion_hidden=128,classifier_hidden=128,dropout=.30,num_classes=8):
        super().__init__(); self.v=BiLSTMBranch(video_dim,proj_dim,branch_hidden,dropout); self.a=BiLSTMBranch(audio_dim,proj_dim,branch_hidden,dropout); self.f=nn.LSTM(4*branch_hidden,fusion_hidden,batch_first=True,bidirectional=True); self.c=nn.Sequential(nn.Dropout(dropout),nn.Linear(2*fusion_hidden,classifier_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(classifier_hidden,num_classes))
    def forward(self,x_video,x_audio): return self.c(self.f(torch.cat([self.v(x_video),self.a(x_audio)],-1))[0].mean(1))
