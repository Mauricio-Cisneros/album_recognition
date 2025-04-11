from picamera2 import Picamera2
import cv2
import numpy as np
import os

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()

folder_path = "/home/mauriciocisneros/Documents/GitHub/album_recognition/album_covers"

# Album metadata dictionary
album_info = {
    "Ali": {
        "title": "Ali",
        "artist": "Khruangbin & Vieux Farka Touré",
        "year": 2022,
        "genre": "Afrobeat / Psychedelic Rock",
        "length": "41:04"
    },
    "Antisocialites": {
        "title": "Antisocialites",
        "artist": "Alvvays",
        "year": 2017,
        "genre": "Indie Pop / Dream Pop",
        "length": "32:38"
    },
    "Fragile": {
        "title": "Fragile",
        "artist": "Yes",
        "year": 1972,
        "genre": "Progressive Rock",
        "length": "37:51"
    },
    "Live in San Francisco '16": {
        "title": "Live in San Francisco '16",
        "artist": "King Gizzard & The Lizard Wizard",
        "year": 2020,
        "genre": "Psychedelic Rock / Garage Rock",
        "length": "1:15:18"
    },
    "Minecraft Volume Alpha": {
        "title": "Minecraft Volume Alpha",
        "artist": "C418",
        "year": 2011,
        "genre": "Ambient / Electronic",
        "length": "1:00:38"
    },
    "Texas Sun": {
        "title": "Texas Sun",
        "artist": "Khruangbin & Leon Bridges",
        "year": 2020,
        "genre": "Soul / Psychedelic Rock",
        "length": "20:13"
    },
    "Things Take Time Take time": {
        "title": "Things Take Time, Take Time",
        "artist": "Courtney Barnett",
        "year": 2021,
        "genre": "Indie Rock / Singer-Songwriter",
        "length": "36:35"
    },
    "This Is Happening": {
        "title": "This Is Happening",
        "artist": "LCD Soundsystem",
        "year": 2010,
        "genre": "Dance-Punk / Electronic",
        "length": "65:05"
    },
    "Wide Awake": {
        "title": "Wide Awake!",
        "artist": "Parquet Courts",
        "year": 2018,
        "genre": "Post-Punk / Indie Rock",
        "length": "38:25"
    }
}

# Load album covers
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


# Setup
album_covers, orb = load_album_covers(folder_path)
show_info = False
locked_album = None

print("Press 'i' to toggle info mode. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()

    if show_info and locked_album:
        matched_album = locked_album
    else:
        matched_album = match_album(frame, album_covers, orb)

    if matched_album:
        cv2.putText(frame, f"Matched: {matched_album}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if show_info and matched_album in album_info:
            info = album_info[matched_album]
            y = 80
            for line in [
                f"Title: {info['title']}",
                f"Artist: {info['artist']}",
                f"Year: {info['year']}",
                f"Genre: {info['genre']}",
                f"Length: {info['length']}"
            ]:
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25

    cv2.imshow("Album Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        show_info = not show_info
        if show_info and matched_album:
            locked_album = matched_album
        else:
            locked_album = None

cv2.destroyAllWindows()
