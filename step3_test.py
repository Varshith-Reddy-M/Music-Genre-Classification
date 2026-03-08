import pickle
import numpy as np
import librosa
import tensorflow as tf
from config import MODEL_SAVE_PATH, SCALER_PATH, GENRES

model = tf.keras.models.load_model(MODEL_SAVE_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
def extract_features(wav_path):
    audio, sr = librosa.load(wav_path, duration=30, mono=True)
    features = []
    features += [len(audio)]
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features += [chroma.mean(), chroma.var()]
    rms = librosa.feature.rms(y=audio)
    features += [rms.mean(), rms.var()]
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features += [sc.mean(), sc.var()]
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features += [sb.mean(), sb.var()]
    ro = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features += [ro.mean(), ro.var()]
    zcr = librosa.feature.zero_crossing_rate(audio)
    features += [zcr.mean(), zcr.var()]
    harmony, perceptr = librosa.effects.hpss(audio)
    features += [harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var()]
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features += [float(tempo)]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    for i in range(20):
        features += [mfcc[i].mean(), mfcc[i].var()]
    return np.array(features, dtype=np.float32).reshape(1, -1)
def predict(wav_path):
    features        = extract_features(wav_path)
    features        = scaler.transform(features)
    probs           = model.predict(features)
    predicted_genre = GENRES[np.argmax(probs)]
    confidence      = float(np.max(probs)) * 100
    print(f"  GENRE      : {predicted_genre.upper()}")
    print(f"  CONFIDENCE : {confidence:.1f}%")
while True:
    path = input("Enter wav file path (or 'exit'): ").strip().strip('"').strip("'")
    if path.lower() == "exit":
        break
    predict(path)