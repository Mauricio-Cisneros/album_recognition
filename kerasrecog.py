import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from picamera2 import Picamera2

# Load the pre-trained model (MobileNetV2 for feature extraction)
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Set path to album cover database
FOLDER_PATH = "/home/mauricio/album_covers_keras"

# Load Raspberry Pi camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Function to extract features from an image using MobileNetV2
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Load and process all album covers into feature vectors
album_features = {}
for album in os.listdir(FOLDER_PATH):
    album_path = os.path.join(FOLDER_PATH, album)
    if os.path.isfile(album_path):
        album_features[album] = extract_features(album_path)

# Function to recognize album covers from camera feed
def recognize_album(input_frame):
    # Convert camera frame for model processing
    frame_resized = cv2.resize(input_frame, (224, 224))
    frame_array = img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = preprocess_input(frame_array)
    
    # Extract features from camera feed
    frame_features = model.predict(frame_array).flatten()

    best_match = None
    best_score = -1

    # Compare against stored album features
    for album_name, album_feature in album_features.items():
        similarity = cosine_similarity([frame_features], [album_feature])[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = album_name

    return best_match, best_score

# Main loop: Process camera feed
while True:
    frame = picam2.capture_array()

    best_match, best_score = recognize_album(frame)

    # Display match result if confidence is high
    if best_score > 0.8:  # Adjust threshold if needed
        print(f"Recognized Album: {best_match} (Confidence: {best_score:.2f})")
        cv2.putText(frame, f"Album: {best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Album Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
