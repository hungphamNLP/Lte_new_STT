import tensorflow as tf
import pickle
import numpy as np
import ffmpeg


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _load_pickle(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj

class Model_Whisper_infer:
    def __init__(self, path_model,feature_extractor,tokenize):
       self.path_model = path_model
       self.feature = feature_extractor
       self.tokenize = tokenize
       self.tflite_gen = self.load_model()
    def load_model(self):
        interpreter = tf.lite.Interpreter(self.path_model)
        tflite_generate = interpreter.get_signature_runner()
        return tflite_generate
    def infer(self,audio,sample_rate=16000):
        inputs = self.feature(audio, sampling_rate=sample_rate, return_tensors="tf")['input_features']
        generated_ids = self.tflite_gen(input_features=inputs)["sequences"]
        transcription = self.tokenize.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription





if __name__ == '__main__':
    filename ='./samples/jfk.wav'
    data_ipt=load_audio(filename)
    #print(data_ipt)

    tflite_model_path = './checkpoint/whisper.tflite'

    #load feature
    feature_extractor = _load_pickle('processor/features_ipt.pkl')

    #load tokenize
    tokenizer_ = _load_pickle('processor/tokenize.pkl')

    model = Model_Whisper_infer(tflite_model_path,feature_extractor,tokenizer_)
    
    opt=model.infer(data_ipt)
    print(opt)
