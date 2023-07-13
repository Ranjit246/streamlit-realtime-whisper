import streamlit as st
import io
import os
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
import tempfile
from time import sleep
from sys import platform
import librosa

from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
#Add language adapters for Odia ('ory')
processor.tokenizer.set_target_lang("ory")
model.load_adapter("ory")

# Session state
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False

# Audio parameters

ENERGY_THRESHOLD = 2000
RECORD_TIMEOUT = 2.0
PHRASE_TIMEOUT = 3.0

args = {
    'model': 'Ranjit/odia_whisper_small_v3.0',
    'non_english': False,
    'energy_threshold': ENERGY_THRESHOLD,
    'record_timeout': RECORD_TIMEOUT,
    'phrase_timeout': PHRASE_TIMEOUT,
}

# The last time a recording was retrieved from the queue.
phrase_time = None
# Current raw audio bytes.
last_sample = bytes()
# Thread-safe Queue for passing data from the threaded recording callback.
data_queue = Queue()
# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = args['energy_threshold']
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

# Select microphone based on platform
if platform == 'linux' or platform == 'linux2':
    mic_name = args['default_microphone']
    if not mic_name or mic_name == 'list':
        st.write("Available microphone devices are:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            st.write(f"Microphone with name \"{name}\" found")
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
else:
    source = sr.Microphone(sample_rate=16000)

temp_file = tempfile.NamedTemporaryFile().name
transcription = ['']

with source as audio_source:
    recorder.adjust_for_ambient_noise(audio_source)


def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push them into the thread-safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)


# Create a background thread that will pass us raw audio bytes.
# We could do this manually, but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=args['record_timeout'])

# Cue the user that we're ready to go.
st.write("Model loaded.\n")

def transcribe(audio):
    y,_ = librosa.load(audio)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    ids = torch.argmax(outputs, dim=-1)[0]
    text = processor.decode(ids)
    return text

def send_receive():
    global phrase_time, last_sample, transcription

    while st.session_state['run']:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=args['phrase_timeout']):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to WAV data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write WAV data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                text = transcribe(temp_file).strip()
                #print(text)

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise, edit the existing one.
                if phrase_complete:
                    transcription[-1] = text
                else:
                    transcription[-1] = text

                # Update session state text
                st.session_state['text'] = '\n'.join(transcription)

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name == 'nt' else 'clear')
                st.write('\n'.join(transcription))
                # Flush stdout.
                st.write('', end='', flush=True)

                # Infinite loops are bad for processors; must sleep.
                sleep(0.25)

        except KeyboardInterrupt:
            break


# Web user interface
st.title('🎙️ Real-Time Transcription App')

with st.expander('About this App'):
    st.markdown('''
    This Streamlit app performs real-time transcription using the Whisper ASR model.
    
    Libraries used:
    - `streamlit` - web framework
    - `speech_recognition` - library for speech recognition and audio recording
    - `torch` - PyTorch library for deep learning
    - `transformers` - library for state-of-the-art NLP models
    ''')

col1, col2 = st.columns(2)

col1.button('Start', on_click=lambda: st.session_state.update({'run': True}))
col2.button('Stop', on_click=lambda: st.session_state.update({'run': False}))

send_receive()
