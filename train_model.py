import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Feature Extraction Function
def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        # Extract features
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. Load Data
csv_file = 'music_data.csv'
df = pd.read_csv(csv_file)

features = []
labels = []

print("Extracting features... The system will now look inside your subfolders.")

for index, row in df.iterrows():
    # constructs path: static/songs/sinhala/calm/song.mp3
    # Make sure folder names in your computer match the CSV 'language' and 'label' exactly!
    file_path = os.path.join('static', 'songs', row['language'], row['label'], row['filename'])
    
    # Check if file exists before trying to load
    if os.path.exists(file_path):
        data = extract_features(file_path)
        if data is not None:
            features.append(data)
            labels.append(row['label'])
    else:
        print(f"File Not Found: {file_path}")
        print(f"--> Check if folder '{row['language']}' or '{row['label']}' exists and matches CSV spelling.")

# 3. Check if we found data
if len(features) == 0:
    print("CRITICAL ERROR: No songs were successfully processed.")
    print("Please check that your folder names in 'static/songs' match the 'language' and 'label' columns in your CSV.")
    exit()

# 4. Train Model
X = np.array(features)
y = np.array(labels)

# If we have very few songs, we might not be able to split train/test. 
# This handles that case for small datasets.
if len(X) < 5:
    print("Warning: Very few songs found. Training on all data without testing split.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 5. Save Model
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model, 'model/music_classifier.pkl')
print("Model saved successfully to model/music_classifier.pkl")