import whisper
import time
import librosa
import torch

UTF8_CHARS = []
for codepoint in range(0x110000):  # Unicode va de U+0000 à U+10FFFF
    try:
        character = chr(codepoint)
        utf8_encoded = character.encode('utf-8')
        UTF8_CHARS.append(character)
    except (ValueError, UnicodeEncodeError):
        pass  

VALID_CHAR = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s"
              "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "'"]

INVALID_CHARS = [i for i in UTF8_CHARS if not(i in VALID_CHAR)]

DECODING_CONFIG = {
    'language':'fr',
    'suppress_token':INVALID_CHARS,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KEEP_CHARS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", 
              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "ä", "â", "à", "ç", "é", "è", "ë", "ê", "ï", "î", "ö", "ô", "ù", "ü", "û", "!", 
              "'", "+", ",", "-", ".", "=", "?", 
              "α", "β",  "γ",  "δ",  "ε",  "ζ",  "η",  "θ",  "ι",  "κ",  "λ",  "μ",  "ν",  "ξ",  "ο",  "π",  "ρ",  "ς",  "σ",  "τ",  "υ",  "φ",
            "χ",  "ψ",  "ω",  "Α", "Β",  "Γ",  "Δ",  "Ε",  "Ζ",  "Η",  "Θ",  "Ι",  "Κ",  "Λ",  "Μ",  "Ν",  "Ξ",  "Ο",  "Π",  "Ρ",  "Σ",  "Τ",  "Υ",  "Φ",  "Χ",  "Ψ",  "Ω"]


class WhisperModel():
    
    def __init__(self, mod="base"):

        # loading model
        self.__model = whisper.load_model(mod)


    def predict(self, audio_path, return_RTF=False, normalize=True):
        if return_RTF :
            duration = librosa.get_duration(audio_path)
            t = time.time()
            res = self.__model.transcribe(audio_path, **DECODING_CONFIG)['text']
            rtf = (time.time() - t)/duration
            if normalize : res = WhisperModel.normalize(res)
            return res, rtf
        return self.__model.transcribe(audio_path, **DECODING_CONFIG)['text']
    
    @staticmethod 
    def normalize(sentence:str):
        res = sentence
        for chr in set(sentence) : 
            if not(chr in KEEP_CHARS) : 
                res = res.replace(chr, '')
        return " ".join(res.split())
