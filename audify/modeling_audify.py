import os
import torch
import warnings
from typing import Dict
from speechbrain.pretrained import EncoderClassifier
# from utils import DEFAULT_TEMP_DIR

warnings.filterwarnings('ignore')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(ROOT_DIR)
# _DEFAULT_CHECKPOINT = '/content/drive/MyDrive/audify/inference'
_DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, 'ckpt')
_DEFAULT_SAVE_PATH = os.path.join(ROOT_DIR, 'ckpt')

# print(_DEFAULT_SAVE_PATH)

class AudifyModel(torch.nn.Module):
    _cache: Dict[str, torch.nn.Module] = {}

    def __init__(self, model_path, save_path):
        super(AudifyModel, self).__init__()
        self.model_path = model_path
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = EncoderClassifier.from_hparams(
            source=model_path, savedir=save_path, run_opts={"device": self.device}
        )

    @classmethod
    def load(cls, model_path=_DEFAULT_CHECKPOINT, save_path=_DEFAULT_SAVE_PATH, cache_model=True):
        """
        In some instances you may want to load the same model twice
        This factory provides a cache so that you don't actually have to load the model twice.
        """
        if model_path in cls._cache:
            return cls._cache[model_path]

        model = AudifyModel(model_path, save_path)
        if cache_model:
            cls._cache[model_path] = model
        return model

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classifier(wavs, wav_lens)

    def classify_file(self, file_path):
        out_prob, score, index, text_lab = self.classifier.classify_file(file_path)

        return out_prob, score, index, text_lab


if __name__ =='__main__':
    import time
    
    model = AudifyModel(_DEFAULT_CHECKPOINT,_DEFAULT_SAVE_PATH)
    model.eval()
    input  = torch.rand(size=[2,1600],dtype=torch.float32,requires_grad=False)
    start=time.time()
    out = model(input)
    end = time.time()
    print(end-start)
    print(out)    