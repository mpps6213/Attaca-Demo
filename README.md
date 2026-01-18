# Riley’s Rhythms — Emotion-Driven Music Experience

**Riley’s Rhythms** is an AI-powered interactive web application that listens to a user’s voice, understands their emotional state, and recommends music or videos that match their current mood — inspired by Disney’s famous movie: *Inside Out*.

Designed for self emotional wellness, Riley’s Rhythms blends **speech recognition**, **natural language processing**, and **emotion-aware UI design** to create a personalized, immersive experience.

— Key Features
 Real-time Voice Input
 Captures live speech through the user’s microphone.


Emotion Detection
 Uses a transformer-based NLP model to classify emotions including:


Joy


Sadness


Anger


Fear


Surprise


Disgust


Neutral


 Dynamic Emotion-Based UI
 The interface dynamically adapts its:


Color theme


Emotion character visuals


Mood indicators
 based on the detected emotion.


 Personalized Media Recommendations


User-selected music genre support, including:
 Pop, Rock, Hip-Hop, R&B, EDM, Dance, Jazz, Classical, K-Pop, Lofi, Indie, Metal, Country, Blues, Reggae, Punk


Spotify mode: Emotion-aware music recommendations tailored to both mood and genre


YouTube mode: Mood- and genre-based music video search links
---

## Tech Stack
UI Framework: Streamlit


Real-Time Speech Recognition: Vosk


Emotion Analysis (NLP): Hugging Face Transformers


Microphone Audio Input: SoundDevice


Music Recommendation Integration: Spotipy (Spotify Web API)


Video Discovery: YouTube Search


Local Asset Handling: Base64 Encoding


Programming Language: Python 3
---

##  Project Structure

ATTACCA_FINAL/
│
├── app.py                  # Main Streamlit UI & application logic
├── requirements.txt        # Python dependencies (Streamlit, Vosk, Transformers, etc.)
├── README.md               # Project documentation
│
├── models/                 # Speech & NLP models (Vosk acoustic model)
│   └── vosk-model-en-us-0.22/
│
├── images/                 # Emotion character assets (UI visuals)
│   ├── joy.png
│   ├── sadness.png
│   ├── anger.png
│   ├── disgust.png
│   ├── fear.png
│   ├── neutral.png
│   └── surprise.png

│
├── RileysRhythms/        ← Virtual Environment (venv)
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── pyvenv.cfg
│
└── __pycache__/            # Python cache files (auto-generated)

##  How to Run the App (Local)

## 1️. Create & activate a virtual environment
```bash
python -m venv venv .\venv\Scripts\Activate
Windows : .\venv\Scripts\Activate
2. Install dependencies : pip install -r requirements.txt
3. Run the application : python -m streamlit run app.py
The website will automatically open in your default web browser.
## Why Riley’s Rhythms? 
Riley’s Rhythms explores how emotion-aware AI can:
Enhance personalization


Improve emotional engagement


Create empathetic human–computer interactions
Potential extensions include:
Mental wellness support tools


Smart entertainment systems


Emotion-adaptive learning platforms
## Team & Hackathon Context
This project was developed for a hackathon to demonstrate:
Applied AI and NLP techniques


Real-time human–AI interaction


Creative UI/UX driven by emotional intelligence


## Notes for Judges 
Speech recognition works offline


No personal data is stored


Designed with clarity, creativity, and technical feasibility in mind


Thank you for exploring Riley’s Rhythms 🎶🧠
