import os
import torch
import warnings
from typing import Dict


_DEFAULT_feature = os.path.join('./', 'audify/script/feature.pth')
_DEFAULT_mean_feature= os.path.join('./', 'audify/script/mean_feature.pth')
_DEFAULT_embedding = os.path.join('./', 'audify/script/embedding.pth')
_DEFAULT_classifier = os.path.join('./', 'audify/script/classifier.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudifyModel(torch.nn.Module):
    def __init__(self):
        super(AudifyModel,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature = torch.jit.load(_DEFAULT_feature)
        self.mean = torch.jit.load(_DEFAULT_mean_feature)
        self.embedding = torch.jit.load(_DEFAULT_embedding)
        self.classifier = torch.jit.load(_DEFAULT_classifier)
    
    def forward(self,wavs):
        if(len(wavs.shape)==1):
            wavs = wavs.unsqueeze(0)
        wav_lens = torch.ones(wavs.shape[0], device=self.device)
        feats = self.feature(wavs)
        feats = self.mean(feats,wav_lens)
        emb = self.embedding(feats,wav_lens)
        out_prob = self.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        return out_prob, score, index
    

if __name__ =='__main__':
    import time
    
    model = AudifyModel()
    model.to(device)
    model.eval()
    input  = torch.rand(size=[2,1600],dtype=torch.float32,device=device,requires_grad=False)
    start=time.time()
    out = model(input)
    end = time.time()
    print(end-start)
    print(out)    