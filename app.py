import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Emotion Detection App", layout="wide")
st.title("😊 Real-Time Emotion Detection System")
st.write("Detect emotions in **real-time**, from **uploaded images** or **videos** using Deep Learning!")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_emotion_model():
    model = load_model("best_emotion_model.keras")  # your trained model file
    return model

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------ LOAD FACE DETECTOR ------------------
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------ EMOTION DETECTION FUNCTION ------------------
def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[preds.argmax()]
        emotions.append(label)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, emotions


# ------------------ PLOT EMOTION GRAPH ------------------
def plot_emotion_distribution(emotion_counts, title):
    if emotion_counts:
        df = pd.DataFrame(list(emotion_counts.items()), columns=["Emotion", "Count"])
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x="Emotion", y="Count", data=df, palette="coolwarm", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        st.pyplot(fig)


# ------------------ APP LAYOUT ------------------
tab1, tab2, tab3 = st.tabs(["📸 Live Webcam", "🖼️ Image Upload", "🎞️ Video Upload"])

# ------------------ TAB 1: LIVE WEBCAM ------------------
with tab1:
    st.subheader("Real-Time Emotion Detection via Webcam")

    start_button = st.button("▶️ Start Webcam")
    stop_button = st.button("⏹️ Stop Webcam")

    FRAME_WINDOW = st.image([])
    emotion_counts = {}

    if start_button:
        cap = cv2.VideoCapture(0)
        st.info("Webcam started. Click 'Stop Webcam' to end.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if stop_button:
                st.warning("Webcam stopped.")
                break

            frame, emotions = detect_emotions(frame)

            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        cv2.destroyAllWindows()

        plot_emotion_distribution(emotion_counts, "Emotion Distribution (Live)")


# ------------------ TAB 2: IMAGE UPLOAD ------------------
with tab2:
    st.subheader("Emotion Detection from Uploaded Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        processed_frame, emotions = detect_emotions(image)
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Detected Emotions", use_container_width=True)

        if emotions:
            st.write("**Detected Emotions:**", ", ".join(emotions))
            emotion_counts = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            plot_emotion_distribution(emotion_counts, "Emotion Distribution (Image)")


# ------------------ TAB 3: VIDEO UPLOAD ------------------
with tab3:
    st.subheader("Emotion Detection from Uploaded Video")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])
        emotion_counts = {}

        st.info("Analyzing video... please wait or click 'Stop Video'.")
        stop_video = st.button("⏹️ Stop Video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_video:
                break

            frame, emotions = detect_emotions(frame)
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        cv2.destroyAllWindows()

        plot_emotion_distribution(emotion_counts, "Emotion Distribution (Video)")
