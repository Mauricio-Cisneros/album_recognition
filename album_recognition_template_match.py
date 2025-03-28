import cv2
import numpy as np
import os
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()


# Set the path where album cover images are stored
FOLDER_PATH = "/home/mauricio/album_recognition/album_covers"

# Function to load album cover images
def load_album_covers(folder_path):
    album_covers = {}
    for album in os.listdir(folder_path):
        album_path = os.path.join(folder_path, album)
        if os.path.isfile(album_path):
            img = cv2.imread(album_path, 0)  # Load in grayscale
            album_covers[album] = img  # Store image with its filename as key
    return album_covers

# Load all album covers
album_covers = load_album_covers(FOLDER_PATH)

# Function to match album cover using Template Matching
def match_album_cover(input_frame, album_covers):
    best_match = None
    best_score = 0
    best_location = None

    # Convert the input frame to grayscale
    frame_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)  # Normalize brightness

    for album_name, template in album_covers.items():
        for scale in np.linspace(0.5, 1.5, 5):  # Try different scales
            resized_template = cv2.resize(template, None, fx=scale, fy=scale)

            if frame_gray.shape[0] < resized_template.shape[0] or frame_gray.shape[1] < resized_template.shape[1]:
                continue  # Skip if the template is bigger than the frame

            result = cv2.matchTemplate(frame_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:  # If a better match is found
                best_score = max_val
                best_match = album_name
                best_location = max_loc

    return best_match, best_score, best_location

# Main Loop: Process Camera Feed
while True:
    frame = picam2.capture_array()

    best_match, best_score, location = match_album_cover(frame, album_covers)

    # Display match result
    if best_match and best_score > 0.8:  # Adjust threshold if needed
        print(f"Recognized Album: {best_match} (Score: {best_score:.2f})")
        cv2.putText(frame, f"Album: {best_match}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Album Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
