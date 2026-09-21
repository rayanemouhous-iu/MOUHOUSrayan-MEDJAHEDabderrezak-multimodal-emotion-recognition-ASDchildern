import numpy as np
import opensmile

class OpenSMILEExtractor:
    def __init__(self):
        self.smile=opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    def extract(self,wav_path,seq_len=40):
        x=self.smile.process_file(wav_path).to_numpy().astype(np.float32)
        if x.ndim!=2 or x.shape[1]!=25: raise ValueError(f'Expected [T,25], got {x.shape}')
        old=np.linspace(0,1,len(x)); new=np.linspace(0,1,seq_len)
        return np.stack([np.interp(new,old,x[:,j]) for j in range(25)],axis=1).astype(np.float32)
