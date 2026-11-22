
import cv2
import torch
import sys
from pathlib import Path
from ultralytics import YOLO


# --------------------------------
# Locate project folders
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

video_path = PROJECT_ROOT / "input" / "movies" / "test_movie.mp4"

output_folder = PROJECT_ROOT / "temp"
output_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------
# Select GPU or CPU
# --------------------------------

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", "GPU" if device == 0 else "CPU")

if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------
# Load YOLO model
# --------------------------------

print("\nLoading YOLO model...")

model = YOLO("yolo11n.pt")


# --------------------------------
# Open the movie
# --------------------------------

video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    print("Error: Could not open the movie.")
    sys.exit(1)


# --------------------------------
# Read frame at 5 seconds
# --------------------------------

video.set(cv2.CAP_PROP_POS_MSEC, 5000)

success, frame = video.read()

if not success:
    print("Error: Could not read video frame.")
    video.release()
    sys.exit(1)


# --------------------------------
# Detect people
# --------------------------------

print("\nDetecting people...")

results = model.predict(
    source=frame,
    classes=[0],       # Class 0 = person
    conf=0.25,
    device=device,
    verbose=False
)

result = results[0]
people_count = 0

# Process every detected person
for index, box in enumerate(result.boxes):

    # Get bounding-box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # Get detection confidence
    confidence = float(box.conf[0])

    # Calculate dimensions and area
    width = x2 - x1
    height = y2 - y1
    area = width * height

    # Calculate the center position
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Count the detected person
    people_count += 1

    # Print detection information
    print(f"\nPerson {index + 1}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Area: {area}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Center: ({center_x}, {center_y})")

    # Draw the bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Display confidence
    cv2.putText(
        frame,
        f"Person {confidence:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    
# --------------------------------
# Select the main person
# --------------------------------

# Store information about detected people
people = []

for box in result.boxes:

    # Get bounding-box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # Get confidence
    confidence = float(box.conf[0])

    # Calculate bounding-box area
    width = x2 - x1
    height = y2 - y1
    area = width * height

    # Store person information
    people.append({
        "box": (x1, y1, x2, y2),
        "area": area,
        "confidence": confidence
    })


# Find the largest detected person
largest_area = max(
    person["area"] for person in people
)

# Calculate a score for each person
for person in people:

    area_score = person["area"] / largest_area

    confidence_score = person["confidence"]

    person["score"] = (
        0.7 * area_score +
        0.3 * confidence_score
    )


# Select the person with the highest score
if people:

    main_person = max(
        people,
        key=lambda person: person["score"]
    )

    print("\nMain Person Selected!")
    print("Area:", main_person["area"])
    print("Confidence:", round(main_person["confidence"], 2))
    print("Score:", round(main_person["score"], 3))
    print("Bounding Box:", main_person["box"])
# --------------------------------
# Save detection result
# --------------------------------

output_path = output_folder / "yolo_detection_test.jpg"

cv2.imwrite(str(output_path), frame)

video.release()


# --------------------------------
# Display results
# --------------------------------

print("\nDetection completed!")
print("People detected:", people_count)
print("Saved image:", output_path)
