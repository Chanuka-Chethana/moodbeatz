from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
import os
import cv2
import numpy as np
import base64
# We use DeepFace now instead of FER
from deepface import DeepFace

app = Flask(__name__)

MUSIC_DATA = 'music_data.csv'

def get_songs_by_emotion(emotion):
    try:
        df = pd.read_csv(MUSIC_DATA)
        matched_songs = df[df['label'].str.lower() == emotion.lower()]
        
        if matched_songs.empty:
            return []
        
        return matched_songs.to_dict(orient='records')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

# --- MANUAL SELECTION ---
@app.route('/get_music', methods=['POST'])
def get_music():
    user_emotion = request.form.get('emotion')
    return find_and_return_song(user_emotion)

# --- WEBCAM AI SCAN ---
@app.route('/scan_emotion', methods=['POST'])
def scan_emotion():
    try:
        data = request.json
        image_data = data['image']

        # 1. Decode the image
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Use DeepFace to analyze
        # enforce_detection=False prevents crash if it can't see a face clearly
        predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        # DeepFace returns a list, we take the first face found
        emotion = predictions[0]['dominant_emotion']
        print(f"AI Detected: {emotion}")

        # 3. Map AI emotions to Music Categories
        # AI sees: angry, disgust, fear, happy, sad, surprise, neutral
        mapped_mood = "calm" # Default

        if emotion in ['happy', 'surprise']:
            mapped_mood = 'happy'
        elif emotion in ['sad', 'fear']:
            mapped_mood = 'sad'
        elif emotion in ['angry', 'disgust']:
            mapped_mood = 'energetic'
        elif emotion == 'neutral':
            mapped_mood = 'calm'

        return find_and_return_song(mapped_mood)

    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'error': 'Could not analyze face. Please try again.'})

def find_and_return_song(mood):
    songs = get_songs_by_emotion(mood)
    
    if not songs:
        # Fallback if no songs exist for that mood
        return jsonify({'error': f"AI saw '{mood}', but no songs found in folder!"})
    
    selected_song = random.choice(songs)
    file_url = f"/static/songs/{selected_song['language']}/{selected_song['label']}/{selected_song['filename']}"
    
    return jsonify({
        'title': selected_song['filename'],
        'language': selected_song['language'],
        'detected_mood': mood,
        'file_url': file_url
    })

if __name__ == '__main__':
    app.run()