
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


# --------------------------------
# Locate project folders
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

video_path = PROJECT_ROOT / "input" / "movies" / "test_movie.mp4"


# --------------------------------
# Select GPU or CPU
# --------------------------------

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", "GPU" if device == 0 else "CPU")


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
    raise RuntimeError("Could not open the movie.")


# --------------------------------
# Read one frame at 5 seconds
# --------------------------------

video.set(cv2.CAP_PROP_POS_MSEC, 5000)

success, frame = video.read()

video.release()

if not success:
    raise RuntimeError("Could not read the video frame.")
# --------------------------------
# Get original video dimensions
# --------------------------------

frame_height, frame_width = frame.shape[:2]

print("Video width:", frame_width)
print("Video height:", frame_height)



# --------------------------------
# Calculate vertical crop dimensions
# --------------------------------

crop_height = frame_height

crop_width = int(crop_height * 9 / 16)

print("\nVertical crop dimensions:")
print("Crop width:", crop_width)
print("Crop height:", crop_height)
# Keep an untouched copy for image analysis
original_frame = frame.copy()
# --------------------------------
# Detect people
# --------------------------------

print("\nDetecting people...")

results = model.predict(
    source=frame,
    classes=[0],
    conf=0.25,
    device=device,
    verbose=False
)

result = results[0]

print("Detection completed!")
print("People detected:", len(result.boxes))

# --------------------------------
# Draw bounding boxes around people
# --------------------------------

for index, box in enumerate(result.boxes):

    # Get the person's coordinates
    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].tolist()
    )

    # Get detection confidence
    confidence = float(box.conf[0])

    # Draw a green rectangle around the person
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Display the person's number and confidence
    cv2.putText(
        frame,
        f"Person {index + 1} ({confidence:.2f})",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
#
# --------------------------------
# Select the largest detected person
# --------------------------------

people = []

for box in result.boxes:

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].tolist()
    )

    width = x2 - x1
    height = y2 - y1

    area = width * height

   
    # Extract the detected person's image region
    person_region = original_frame[y1:y2, x1:x2]

    # Convert the region into grayscale
    gray_region = cv2.cvtColor(
        person_region,
        cv2.COLOR_BGR2GRAY
    )

    # Measure image sharpness
    sharpness = cv2.Laplacian(
        gray_region,
        cv2.CV_64F
    ).var()

    people.append({
        "box": (x1, y1, x2, y2),
        "area": area,
        "sharpness": sharpness
    })

print("\nPerson sharpness scores:")

for index, person in enumerate(people, start=1):
    print(
        f"Person {index}: "
        f"Area={person['area']}, "
        f"Sharpness={person['sharpness']:.2f}"
    )

if people:

    # Find the maximum values for normalization
    max_area = max(person["area"] for person in people)

    max_sharpness = max(
        person["sharpness"] for person in people
    )

    # Calculate a combined score
    for person in people:

        # Normalize person size
        area_score = person["area"] / max_area

        # Normalize image sharpness
        sharpness_score = (
            person["sharpness"] / max_sharpness
        )

        # Combine both measurements
        person["score"] = (
            0.6 * area_score +
            0.4 * sharpness_score
        )

    # Select the highest-scoring person
    main_person = max(
        people,
        key=lambda person: person["score"]
    )

    x1, y1, x2, y2 = main_person["box"]
    
    # --------------------------------
    # Calculate crop position
    # --------------------------------

    # Find the horizontal centre of the selected person
    person_center_x = (x1 + x2) // 2

    # Place the crop around the person's centre
    crop_x = person_center_x - (crop_width // 2)

    # Prevent the crop from going outside the original frame
    crop_x = max(
        0,
        min(crop_x, frame_width - crop_width)
    )

    print("\nCrop position:")
    print("Crop X:", crop_x)
    print("Crop Y:", 0)

    # Highlight the selected person in red
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        5
    )

    cv2.putText(
        frame,
        "SELECTED PERSON",
        (x1, max(y1 - 20, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    print("\nSelected person:")
    print("Bounding box:", main_person["box"])
    print("Area:", main_person["area"])
    print("Sharpness:", round(main_person["sharpness"], 2))
    print("Combined score:", round(main_person["score"], 3))


    x1, y1, x2, y2 = main_person["box"]

    # Highlight the selected person in red
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        5
    )

    cv2.putText(
        frame,
        "SELECTED PERSON",
        (x1, max(y1 - 20, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )
   
    # --------------------------------
    # Draw the vertical crop window
    # --------------------------------

    crop_x2 = crop_x + crop_width

    cv2.rectangle(
        frame,
        (crop_x, 0),
        (crop_x2, frame_height),
        (255, 0, 0),
        5
    )

    cv2.putText(
        frame,
        "VERTICAL CROP AREA",
        (crop_x + 10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    print("\nSelected person:")
    print("Bounding box:", main_person["box"])
    print("Area:", main_person["area"])


# --------------------------------
# Display the detection result
# --------------------------------

cv2.imshow("YOLO Person Detection", frame)

print("\nPress any key to close the image.")

cv2.waitKey(0)
cv2.destroyAllWindows()
