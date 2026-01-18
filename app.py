import os
import json
import time
import queue
import urllib.parse
import base64

import streamlit as st
import streamlit.components.v1 as components
import sounddevice as sd

from vosk import Model, KaldiRecognizer
from transformers import pipeline

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# -----------------------------
# 1. CONFIG & CONSTANTS
# -----------------------------
VOSK_MODEL_PATH = r"C:\Users\User\Attacca_Final\models\vosk-model-en-us-0.22"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

SAMPLE_RATE = 16000
BLOCKSIZE = 4000

GENRE_OPTIONS = [
    "pop", "rock", "hip-hop", "r-n-b", "edm", "dance",
    "jazz", "classical", "k-pop", "lofi", "indie",
    "metal", "country", "blues", "reggae", "punk"
]

EMOTION_TO_CHARACTER = {
    "joy":      {"name": "Joy",      "emoji": "💛", "color": "#FEE033", "darker": "#998200", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\joy.png"},
    "sadness":  {"name": "Sadness",  "emoji": "💙", "color": "#4A90E2", "darker": "#214a7a", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\sadness.png"},
    "anger":    {"name": "Anger",    "emoji": "🔴", "color": "#E23E28", "darker": "#8b1a0d", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\anger.png"},
    "fear":     {"name": "Fear",     "emoji": "🟣", "color": "#A386D5", "darker": "#5e438a", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\fear.png"},
    "surprise": {"name": "Surprise", "emoji": "🩷", "color": "#FF4DD8", "darker": "#f71ccf", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\surprise.png"},
    "disgust":  {"name": "Disgust",  "emoji": "💚", "color": "#76D672", "darker": "#3a6d38", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\disgust.png"},
    "neutral":  {"name": "Ennui",    "emoji": "⚪", "color": "#808080", "darker": "#404040", "text": "#ffffff", "image": r"C:\\Users\\User\\Attacca_Final\\images\\neutral.png"}
}

# -----------------------------
# 2. HELPER FUNCTIONS
# -----------------------------
def get_local_img(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{data}"
    except Exception as e:
        print(f"Error: {e}")
    return None

@st.cache_resource
def load_vosk():
    if not os.path.exists(VOSK_MODEL_PATH):
        st.error(f"Vosk model not found.")
        return None
    return Model(VOSK_MODEL_PATH)

@st.cache_resource
def load_emotion_clf():
    return pipeline("text-classification", model=EMOTION_MODEL_NAME, top_k=None)

vosk_model = load_vosk()
emotion_clf = load_emotion_clf()

def record_and_transcribe(duration_sec):
    q = queue.Queue()
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))
    transcript_parts = []
    status_placeholder = st.empty()
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype="int16", channels=1, callback=callback):
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            data = q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res['text']: transcript_parts.append(res['text'])
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    status_placeholder.markdown(f"<p style='text-align:center; color:white; font-style:italic;'>🧠 Belief System Updating: {partial}...</p>", unsafe_allow_html=True)
    final = json.loads(rec.FinalResult()).get("text", "")
    if final: transcript_parts.append(final)
    status_placeholder.empty()
    return " ".join(transcript_parts)

# -----------------------------
# 3. UI CONFIG
# -----------------------------
st.set_page_config(page_title="Attacca | Emotions", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .block-container {
        max-width: 100% !important;
        padding: 3rem 5rem !important;
    }

    .stApp { 
        background-color: #2c3e50; 
        transition: background-color 1.5s ease-in-out;
        color: white;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3) !important;
    }

    .main-header { 
        font-size: 4.5rem; 
        font-weight: 900; 
        text-align: center; 
        color: white;
        margin-bottom: 0px; 
    }
    
    .sub-text { 
        text-align: center; 
        color: rgba(255,255,255,0.7);
        font-style: italic;
        font-size: 1.2rem;
        margin-bottom: 40px; 
    }
    
    /* SHIFTING THE ORB TO THE RIGHT */
    div.stButton {
        display: flex;
        justify-content: center;
        width: 100%;
        /* Shifted further right to sit under 'Personal Memory Console' */
        padding-left: 400px !important; 
    }

    div.stButton > button:first-child {
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #e0e0e0 20%, #a0a0a0 100%);
        border-radius: 50% !important;
        width: 220px !important; 
        height: 220px !important;
        border: 8px solid rgba(255, 255, 255, 0.4) !important;
        color: transparent !important;
        text-indent: -9999px;
        box-shadow: inset -10px -10px 40px rgba(0,0,0,0.2), 0 0 30px rgba(255,255,255,0.3);
    }

    div.stButton > button:active, div.stButton > button:focus {
        background: radial-gradient(circle at 30% 30%, #fff9c4 0%, #fbc02d 50%, #f57f17 100%);
        box-shadow: 0 0 50px #fbc02d, 0 0 100px rgba(251, 192, 45, 0.5);
        transform: scale(1.05);
        animation: spin 3s linear infinite;
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .character-img {
        display: block;
        margin: 20px auto;
        width: 250px;
    }

    /* Keep Sidebar Dropdown text color Black */
    [data-testid="stSidebar"] div[data-baseweb="select"] div,
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    platform = st.selectbox("Select Medium", ["Spotify", "YouTube"])
    LOGOS = {"Spotify": "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", "YouTube": "https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_%282017%29.svg"}
    st.image(LOGOS[platform], width=70)
    st.divider()
    genre = st.selectbox("Preferred Genre", GENRE_OPTIONS)
    duration = st.slider("Orb Energy Duration (s)", 3, 10, 5)

# --- MAIN CONTENT ---
st.markdown("<h1 class='main-header'>Riley's Rhythms</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Inside Out 2: Personal Memory Console</p>", unsafe_allow_html=True)

if st.button("RECORD"):
    transcript = record_and_transcribe(duration)
    if transcript:
        scores = emotion_clf(transcript)[0]
        best_emo = max(scores, key=lambda x: x['score'])
        mood = best_emo['label']
        char = EMOTION_TO_CHARACTER.get(mood, EMOTION_TO_CHARACTER["neutral"])
        img_src = get_local_img(char['image'])
        
        st.markdown(f"<style>.stApp {{ background-color: {char['color']} !important; }} .main-header, .sub-text, h3, h4, p {{ color: {char['text']} !important; }}</style>", unsafe_allow_html=True)

        st.markdown(f"<div style='text-align:center;'><h3>“{transcript}”</h3></div>", unsafe_allow_html=True)
        if img_src: st.markdown(f'<img src="{img_src}" class="character-img">', unsafe_allow_html=True)

        confidence = round(best_emo['score'] * 100)
        st.markdown(f"<div style='max-width:600px; margin:auto; text-align:center; padding:15px; background:rgba(255,255,255,0.2); border-radius:15px; color:{char['text']}; border: 2px solid {char['text']};'>Core Memory Formed: {char['emoji']} <b>{char['name'].upper()}</b> ({confidence}% Energy)</div>", unsafe_allow_html=True)
        st.divider()

        # Rec Logic
        media_type = "music" if platform == "Spotify" else "video"
        rec_header = f"{char['name']} recommends this {media_type} to Riley"
        box_style = f"background: linear-gradient(135deg, {char['darker']} 0%, #121212 100%); border: 2px solid {char['color']}; border-radius: 15px; padding: 20px; text-align: center; max-width: 800px; margin: 10px auto;"
        
        if platform == "Spotify":
            st.markdown(f"<div style='{box_style}'><h4 style='color:{char['text']}; text-align:center;'>{rec_header}</h4><a href='https://open.spotify.com/search/{genre}%20{mood}' target='_blank' style='color:{char['color']};'>Open {char['name']}'s Playlist →</a></div>", unsafe_allow_html=True)
        else:
            search_query = urllib.parse.quote(f"{genre} {mood} music video")
            st.markdown(f"<div style='{box_style}'><h3 style='color:{char['color']};'>{rec_header}</h3><a href='https://www.youtube.com/results?search_query={search_query}' target='_blank' style='color:{char['color']}; font-weight:bold;'>Explore on YouTube →</a></div>", unsafe_allow_html=True)

st.markdown("<br><br><p style='text-align:center; opacity:0.6;'>Inside Out 2 Emotion Engine</p>", unsafe_allow_html=True)