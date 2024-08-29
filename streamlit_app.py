import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard as copy
import os
import time

from audiorecorder import audiorecorder
import whisper
from src.FST.transducers import *


c1, c2, c3 = st.columns(3, gap='large')
with c2 :
    st.title(':red[TaL]:blue[eX]')

accepted_models = ['base', 'small', 'medium']

if not 'model' in st.session_state : 
    st.session_state.model = None 

if not 'Lex_FST' in st.session_state :
    st.session_state.Lex_FST = None

if not 'Lex_Gram_FST' in st.session_state :
    st.session_state.Lex_Gram_FST = None

if not 'Lex_Gram4_FST' in st.session_state :
    st.session_state.Lex_Gram4_FST = None


st.write("___")
st.subheader(":orange[Configuration]")

def search_for_models():
    home_path = os.path.expanduser("~")
    cache = os.path.join(home_path, '.cache/whisper')
    return [s.split('.')[0] for s in os.listdir(cache)]

models_types = search_for_models()

def load_model(model_type:str):
    return whisper.load_model(model_type)
  
st.write(":red[whisper config]")

c1, c2 = st.columns(2)

with c1 : 
    select_model = st.selectbox("Whisper model size", accepted_models)

with c2 :
    if st.button('Load model'):
        st.session_state.model = load_model(select_model)

st.write(":red[FST config]")

select_transducer = st.selectbox("Transducer type", ["lex", "lex + gram", "lex + gram4"])

if st.session_state.Lex_FST == None : st.session_state.Lex_FST = LexMathTransducer()
if st.session_state.Lex_Gram_FST == None : st.session_state.Lex_Gram_FST = LexGraOneLayerFST()
if st.session_state.Lex_Gram4_FST == None : st.session_state.Lex_Gram4_FST = LexGraMultiLayerFST()
  
if st.button("Reload FSTs"):
    st.session_state.Lex_FST = LexMathTransducer()
    st.session_state.Lex_Gram_FST = LexGraOneLayerFST()
    st.session_state.Lex_Gram4_FST = LexGraMultiLayerFST()

st.write("___")
st.subheader(":orange[Demo]")

  
audio = audiorecorder("Enregistrer", "Terminer l'enregistrement")

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.export().read())  

    # To save audio to a file, use pydub export method:
    audio.export("audio.wav", format="wav")

    # To get audio properties, use pydub AudioSegment properties:
    st.write(f"Sample rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

    if st.session_state.FST != None and st.session_state.model != None :

        transcript = st.session_state.model.transcribe('audio.wav')['text']
        
        if select_transducer == 'lex' : 

            result =  st.session_state.Lex_FST.predict(transcript)
  
            a, b = st.columns(2)
            with a : 
                st.write('**:green[Prediction Whisper (norm + lemm) :]**')
            with b :
                st.write(st.session_state.Lex_FST.normalized_sequence(transcript))
                
            a, b = st.columns(2)
            with a :  
                st.write('**:green[Sortie LaTeX :]**')
            with b :
                d, e = st.columns(2)
                with d :
                    st.write('$' + result + '$')
                with e :
                    copy(result)
                

        else :
            
            if select_transducer == 'lex + gram' :  result = st.session_state.Lex_Gram_FST.predict(transcript, nbest=5)
            elif select_transducer == 'lex + gram4' : result = st.session_state.Lex_Gram4_FST.predict(transcript, nbest=5)
              
            if len(result) > 0 :
                
                a, b = st.columns(2)
                with a : 
                    st.write('**:green[Prediction Whisper (norm + lemm) :]**')
                with b :
                    st.write(st.session_state.Lex_FST.normalized_sequence(transcript))
                
                a, b = st.columns(2)
                with a :  
                    st.write('**:green[Sortie LaTeX :]**')
                with b :
                    d, e = st.columns(2)
                    with d :
                        st.write('$' + result[0] + '$')
                    with e :
                        copy(result[0])
                    
                a, b = st.columns(2)
                with a :  
                    st.write('**:green[Alternatives :]**')
                with b :
                    for hyp in result[1:]:
                        st.write('$' + hyp + '$')

