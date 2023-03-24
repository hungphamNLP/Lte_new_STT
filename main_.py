from audio import AudioFile
from pathlib import Path
path_root = Path(__file__).parents[0]
from utils import extract_audio,convert_audio,write_to_file,DEFAULT_TEMP_DIR
import tqdm


from model_tf_lite import Model_Whisper_infer,_load_pickle






if __name__ =='__main__':
    tflite_model_path = './checkpoint/whisper.tflite'

    #load feature
    feature_extractor = _load_pickle('processor/features_ipt.pkl')

    #load tokenize
    tokenizer_ = _load_pickle('processor/tokenize.pkl')

    model = Model_Whisper_infer(tflite_model_path,feature_extractor,tokenizer_)
    
    
    allow_tags = {"speech", "male", "female", "noisy_speech", "music"}
    audio_file = AudioFile(audio_path='./samples/jfk.wav')
    transcribe_music=False
    recognize_tokens = []
    for (start,end,audio,tag) in audio_file.split(backend='vad',classify=False):
        if tag not in allow_tags:
            continue
        if tag == "music" and not transcribe_music:
            if trans_dict is not None:
                # final_tokens = self.post_process(trans_dict['tokens'], auto_punc=auto_punc)
                # recognize_tokens.extend(final_tokens)
                trans_dict = None
            recognize_tokens.append(
                {
                    "text": "[âm nhạc]",
                    "start": start,
                    "end": end,
                }
            )
            continue
        print(model.infer(audio))
