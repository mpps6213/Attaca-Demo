import wave
import json
from vosk import Model, KaldiRecognizer
from textblob import TextBlob
from nrclex import NRCLex

# -----------------------------
# STEP 1: Speech Recognition
# -----------------------------
def transcribe_audio(audio_path, model_path="models/vosk-model-small-en-us-0.15"):
    wf = wave.open(audio_path, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())

    transcript = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result['text'] + " "
    return transcript.strip()

# -----------------------------
# STEP 2: Sentiment + Emotion
# -----------------------------
def analyze_emotion(text):
    # TextBlob sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # NRCLex emotions
    emotion = NRCLex(text)
    top_emotions = emotion.top_emotions
    raw_scores = emotion.raw_emotion_scores

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "top_emotions": top_emotions,
        "raw_scores": raw_scores
    }
    

# -----------------------------
# STEP 3: Main Pipeline
# -----------------------------
if __name__ == "__main__":
    audio_file = "data/sample.wav"  # replace with your audio file
    transcript = transcribe_audio(audio_file)
    print("Transcript:", transcript)

    results = analyze_emotion(transcript)
    print("\n--- Analysis ---")
    print("Polarity:", results["polarity"])
    print("Subjectivity:", results["subjectivity"])
    print("Top Emotions:", results["top_emotions"])
    print("All Emotion Scores:", results["raw_scores"])


from transformers import pipeline

# Load the emotion classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def get_emotion(user_text):
    result = classifier(user_text)
    # Returns something like: [{'label': 'joy', 'score': 0.98}]
    return result[0]['label']
