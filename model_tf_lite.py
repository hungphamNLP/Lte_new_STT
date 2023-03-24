import tensorflow as tf
import pickle


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
    from scipy.io import wavfile
    import torchaudio
    filename ='./samples/jfk.wav'
    data,rt = torchaudio.load(filename)
    
    #load model
    tflite_model_path = './checkpoint/whisper.tflite'

    #load feature
    feature_extractor = _load_pickle('processor/features_ipt.pkl')

    #load tokenize
    tokenizer_ = _load_pickle('processor/tokenize.pkl')

    model = Model_Whisper_infer(tflite_model_path,feature_extractor,tokenizer_)
    
    opt=model.infer(data[0])
    print(opt)
