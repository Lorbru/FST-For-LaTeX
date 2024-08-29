from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from transformers import AutoModelForCTC, Wav2Vec2ProcessorWithLM
from src.FST.transducers import Normalizer

import torch
import torchaudio
import librosa 


UTF8_CHARS = []
for codepoint in range(0x110000):  
    try:
        character = chr(codepoint)
        utf8_encoded = character.encode('utf-8')
        UTF8_CHARS.append(character)
    except (ValueError, UnicodeEncodeError):
        pass  

VALID_CHAR = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", 
              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "ä", "â", "à", "ç", "é", "è", "ë", "ê", "ï", "î", "ö", "ô", "ù", "ü", "û", "!", 
              "'", "+", ",", "-", ".", "=", "?", 
              "α", "β",  "γ",  "δ",  "ε",  "ζ",  "η",  "θ",  "ι",  "κ",  "λ",  "μ",  "ν",  "ξ",  "ο",  "π",  "ρ",  "ς",  "σ",  "τ",  "υ",  "φ",
            "χ",  "ψ",  "ω",  "Α", "Β",  "Γ",  "Δ",  "Ε",  "Ζ",  "Η",  "Θ",  "Ι",  "Κ",  "Λ",  "Μ",  "Ν",  "Ξ",  "Ο",  "Π",  "Ρ",  "Σ",  "Τ",  "Υ",  "Φ",  "Χ",  "Ψ",  "Ω"]

INVALID_CHARS = [i for i in UTF8_CHARS if not(i in VALID_CHAR)]


class OpAI_WhisperModel():
    
    def __init__(self, mod="base", lang=None):
        self.__processor = WhisperProcessor.from_pretrained("openai/whisper-"+mod, language=lang)
        self.__model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-"+mod)
        self.__normalizer = Normalizer()

    def predict(self, audio_file, normalize=True):

        # Audio loading
        audio, sample_rate = librosa.load(audio_file, sr=16000)

        # Process audio
        input_features = self.__processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features

        # Model inference
        predicted_ids = self.__model.generate(input_features)

        # Decoder
        transcription = self.__processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Post normalization
        if normalize : 
            transcription = ''.join([c for c in transcription if c in VALID_CHAR])
            transcription = self.__normalizer.predict(transcription)
        
        return transcription
    
class Wav2Vec2_7kLarge_BofengHuangLM_FR():
    
    def __init__(self):

        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(self.__device)
        self.__processor = Wav2Vec2ProcessorWithLM.from_pretrained("bhuang/asr-wav2vec2-french")
        self.__model_sample_rate = self.__processor.feature_extractor.sampling_rate
        self.__normalizer = Normalizer()


    def predict(self, audio_file, normalize=True):

        waveform, sample_rate = torchaudio.load(audio_file) 

        # channels dim
        num_channels = waveform.shape[0]    
        waveform = waveform[num_channels - 1:].squeeze(axis=0)

        # resample
        if sample_rate != self.__model_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.__model_sample_rate)
            waveform = resampler(waveform)

        # normalize
        input_dict = self.__processor(waveform, sampling_rate=self.__model_sample_rate, return_tensors="pt")

        with torch.inference_mode():
            logits = self.__model(input_dict.input_values.to(self.__device)).logits

        # decode
        transcription = predicted_sentence = self.__processor.batch_decode(logits.cpu().numpy()).text[0]

        # Post normalization
        if normalize : 
            transcription = ''.join([c for c in transcription if c in VALID_CHAR])
            transcription = self.__normalizer.predict(transcription)
        
        return transcription
