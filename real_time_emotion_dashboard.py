import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from collections import Counter
import csv
from datetime import datetime

# ----------------------------
# Load model and setup
# ----------------------------
print(" Loading emotion detection model...")
model = load_model("best_emotion_model.keras")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face detector
print("📸 Loading Haar cascade face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: Webcam not accessible!")
    exit()

# Emotion counter for analytics
emotion_counter = Counter()

# ----------------------------
# CSV Logging Setup
# ----------------------------
log_file = open("emotion_log.csv", mode="a", newline="")
writer = csv.writer(log_file)
writer.writerow(["Timestamp", "Emotion", "Confidence"])

# ----------------------------
# Real-Time Chart Setup
# ----------------------------
plt.ion()  # Interactive mode for live updating charts

print(" System ready! Press 'q' to quit.\n")

# ----------------------------
# Real-Time Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        # Resize and normalize face for model
        face_resized = cv2.resize(roi_gray, (48, 48))
        face_norm = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_norm, axis=(0, -1))

        # Predict emotion
        preds = model.predict(face_input, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)

        # Count & log emotion
        emotion_counter[emotion] += 1
        writer.writerow([datetime.now(), emotion, f"{confidence:.2f}"])

        # Draw bounding box & emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ----------------------------
    # Live Analytics Chart
    # ----------------------------
    plt.clf()
    plt.bar(emotion_counter.keys(), emotion_counter.values(), color='skyblue')
    plt.title("Emotion Frequency Over Time")
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.pause(0.01)

    # ----------------------------
    # Show Webcam Feed
    # ----------------------------
    cv2.imshow(" Real-Time Emotion Detection Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
plt.close()
log_file.close()

print(" Emotion log saved to 'emotion_log.csv'")
print(" Program ended successfully.")
