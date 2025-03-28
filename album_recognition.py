from picamera2 import Picamera2
import cv2
import numpy as np
import os

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()


folder_path = "/home/mauricio/album_recognition/album_covers"

# Function to load album covers
def load_album_covers(folder_path):
    orb = cv2.ORB_create()
    album_covers = {}

    for album_name in os.listdir(folder_path):
        album_path = os.path.join(folder_path, album_name)

        if os.path.isdir(album_path):
            album_covers[album_name] = []

            for filename in os.listdir(album_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(album_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None:
                        album_covers[album_name].append((img, kp, des))

    return album_covers, orb


# Function to match album
def match_album(frame, album_covers, orb, threshold=10):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    max_matches = 0

    for album_name, covers in album_covers.items():
        for album_img, kp_album, des_album in covers:
            if des_album is None:
                continue

            matches = bf.match(des_album, des_frame)

            if len(matches) > max_matches and len(matches) > threshold:
                max_matches = len(matches)
                best_match = album_name

    return best_match


# Load album covers
album_covers, orb = load_album_covers(folder_path)

while True:
    
    frame = picam2.capture_array()
    matched_album = match_album(frame, album_covers, orb)

    if matched_album:
        cv2.putText(frame, f"Matched: {matched_album}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Album Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
