# Riley’s Rhythms — Emotion-Driven Music Experience

**Riley’s Rhythms** is an AI-powered interactive web application that listens to a user’s voice, understands their emotional state, and recommends music or videos that match their current mood, inspired by Disney’s *Inside Out*.

Designed to support emotional wellness, Riley’s Rhythms blends **speech recognition**, **natural language processing**, and **emotion-aware UI design** to create a personalized and immersive experience.

---

## Key Features

### Real-Time Voice Input
- Captures live speech through the user’s microphone.

### Emotion Detection
- Uses a transformer-based NLP model to classify emotions, including:
  - Joy  
  - Sadness  
  - Anger  
  - Fear  
  - Surprise  
  - Disgust  
  - Neutral  

### Dynamic Emotion-Based UI
- The interface dynamically adapts its:
  - Color theme  
  - Emotion character visuals  
  - Mood indicators  
- Visual changes reflect the detected emotional state.

### Personalized Media Recommendations
- User-selected music genre support, including:  
  Pop, Rock, Hip-Hop, R&B, EDM, Dance, Jazz, Classical, K-Pop, Lofi, Indie, Metal, Country, Blues, Reggae, Punk
- Spotify mode provides emotion-aware music recommendations tailored to both mood and genre.
- YouTube mode provides mood- and genre-based music video search links.

---

## Tech Stack

- UI Framework: Streamlit  
- Real-Time Speech Recognition: Vosk  
- Emotion Analysis (NLP): Hugging Face Transformers  
- Microphone Audio Input: SoundDevice  
- Music Recommendation Integration: Spotipy (Spotify Web API)  
- Video Discovery: YouTube Search  
- Local Asset Handling: Base64 Encoding  
- Programming Language: Python 3  

---

## Project Structure

ATTACCA_FINAL/
│
├── app.py # Main Streamlit UI & application logic
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── models/ # Speech & NLP models (Vosk acoustic model)
│ └── vosk-model-en-us-0.22/
│
├── images/ # Emotion character assets
│ ├── joy.png
│ ├── sadness.png
│ ├── anger.png
│ ├── disgust.png
│ ├── fear.png
│ ├── neutral.png
│ └── surprise.png
│
├── venv/ # Virtual environment (not committed)
│ ├── Include/
│ ├── Lib/
│ ├── Scripts/
│ └── pyvenv.cfg
│
└── pycache/ # Python cache files (auto-generated)

yaml
Copy code

---

## How to Run the App (Local)

### 1. Create and activate a virtual environment
```bash
python -m venv venv
Windows

powershell
Copy code
.\venv\Scripts\Activate
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the application
bash
Copy code
python -m streamlit run app.py
The website will automatically open in your default web browser.

Why Riley’s Rhythms?
Riley’s Rhythms explores how emotion-aware AI can:

Enhance personalization

Improve emotional engagement

Create empathetic human–computer interactions

Potential extensions include:

Mental wellness support tools

Smart entertainment systems

Emotion-adaptive learning platforms

Team & Hackathon Context
This project was developed for a hackathon to demonstrate:

Applied AI and NLP techniques

Real-time human–AI interaction

Creative UI/UX driven by emotional intelligence

AI Assistance & Acknowledgements
This project was developed with the assistance of AI tools for ideation, debugging, and documentation support:

ChatGPT (OpenAI) — used for code explanation, documentation drafting, and design guidance

Gemini (Google) — used for brainstorming and conceptual refinement

GitHub Copilot — used for code suggestions and productivity support

All final implementation decisions, testing, and integration were performed by the project team.

Notes for Judges
Speech recognition works offline

No personal data is stored

Designed with clarity, creativity, and technical feasibility in mind

Thank you for exploring Riley’s Rhythms.
