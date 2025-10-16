import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained emotion model
model = load_model("best_emotion_model.keras")

# Emotion categories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame skipping for optimization
frame_count = 0

# Start capturing
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (smaller = faster)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle on color frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Crop and preprocess face for emotion model
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_norm, axis=(0, -1))

        # Predict emotion
        prediction = model.predict(face_input, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(prediction)

        # Draw label and confidence on frame
        label_text = f"{emotion_label} ({confidence*100:.1f}%)"
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
     
     
    cv2.namedWindow("Optimized Real-Time Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optimized Real-Time Emotion Detection", 1200, 800)
 
    # Display the result
    cv2.imshow("Optimized Real-Time Emotion Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
