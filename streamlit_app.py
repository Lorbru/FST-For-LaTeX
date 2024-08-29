import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard as copy
import os
import time

from audiorecorder import audiorecorder
import whisper
from src.FST.transducers import *


st.title('Génération de code LaTeX par usage de la parole')

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

def load_model(model_type:str):
    return whisper.load_model(model_type)
  
st.write(":red[whisper config]")

c1, c2 = st.columns(2)

with c1 : 
    select_model = st.selectbox("Whisper model size", accepted_models)

with c2 :
    if st.button('Load ASR'):
        st.session_state.model = load_model(select_model)

st.write(":red[FST config]")

select_transducer = st.selectbox("Transducer type", ["lex", "lex + gram", "lex + gram4"])

if st.session_state.Lex_FST == None : st.session_state.Lex_FST = LexMathTransducer()
if st.session_state.Lex_Gram_FST == None : st.session_state.Lex_Gram_FST = LexGraOneLayerFST()
if st.session_state.Lex_Gram4_FST == None : st.session_state.Lex_Gram4_FST = LexGraMultiLayerFST()
  
if st.button("Load FST"):
    if select_transducer == "lex" : st.session_state.Lex_FST = LexMathTransducer()
    elif select_transducer == "lex + gram" : st.session_state.Lex_Gram_FST = LexGraOneLayerFST()
    elif select_transducer == "lex + gram4" : st.session_state.Lex_Gram4_FST = LexGraMultiLayerFST()

if select_transducer == "lex" : FST = st.session_state.Lex_FST 
elif select_transducer == "lex + gram" : FST = st.session_state.Lex_Gram_FST
elif select_transducer == "lex + gram4" : FST = st.session_state.Lex_Gram4_FST

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

    if FST != None and st.session_state.model != None :

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

    else :

        st.write(":red[>> Choisir un modèle ASR")

