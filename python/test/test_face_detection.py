
import cv2
import sys
from pathlib import Path


# --------------------------------
# Locate the main Python folder
# --------------------------------

PYTHON_FOLDER = Path(__file__).resolve().parent.parent

# Allow Python to import face_detector.py
sys.path.insert(0, str(PYTHON_FOLDER))

from face_detector import detect_faces


# --------------------------------
# Load our test movie
# --------------------------------

PROJECT_ROOT = PYTHON_FOLDER.parent

video_path = PROJECT_ROOT / "input" / "movies" / "test_movie.mp4"

# Open the video
video = cv2.VideoCapture(str(video_path))


# Check whether the video opened
if not video.isOpened():

    print("Error: Could not open the video.")
    exit()


# --------------------------------
# Select a frame at 5 seconds
# --------------------------------

video.set(cv2.CAP_PROP_POS_MSEC, 5000)

# Read the selected frame
success, frame = video.read()


if not success:

    print("Error: Could not read the video frame.")

else:

    # --------------------------------
    # Detect faces
    # --------------------------------

    faces = detect_faces(frame)

    print("Faces detected:", len(faces))

    # Draw rectangles around detected faces
    for (x, y, width, height) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        print(
            f"Face position: x={x}, y={y}, "
            f"width={width}, height={height}"
        )

    # --------------------------------
    # Save the test image
    # --------------------------------

    output_folder = PROJECT_ROOT / "temp"

    # Create the folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "face_detection_test.jpg"

    # Save the frame with detected faces
    cv2.imwrite(str(output_path), frame)

    print("\nFace detection test completed!")
    print("Test image saved to:", output_path)


# Release the video resource
video.release()
