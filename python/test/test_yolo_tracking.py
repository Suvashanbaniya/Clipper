
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
# Store tracking statistics
# --------------------------------

track_stats = {}
# --------------------------------
# Start person tracking
# --------------------------------

print("\nStarting person tracking...")

while video.isOpened():

    success, frame = video.read()

    if not success:
        break

    results = model.track(
        source=frame,
        persist=True,
        classes=[0],
        conf=0.25,
        tracker="bytetrack.yaml",
        device=device,
        verbose=False
    )

    result = results[0]

    # Check whether tracking IDs are available
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, confidence in zip(
            boxes, track_ids, confidences
        ):
            x1, y1, x2, y2 = map(int, box)

            # Calculate the person's bounding-box area
            area = (x2 - x1) * (y2 - y1)

            # Create a record for a new tracking ID
            if track_id not in track_stats:
                track_stats[track_id] = {
                    "total_area": 0,
                    "total_confidence": 0,
                    "frames_seen": 0
                }

            # Update the person's statistics
            track_stats[track_id]["total_area"] += area
            track_stats[track_id]["total_confidence"] += confidence
            track_stats[track_id]["frames_seen"] += 1

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Draw the person's bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Display tracking ID
            cv2.putText(
                frame,
                f"Person ID: {track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display the tracking result
    cv2.imshow("YOLO Person Tracking", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# Release resources
# --------------------------------

video.release()
cv2.destroyAllWindows()



# --------------------------------
# Calculate tracking statistics
# --------------------------------

person_rankings = []

for track_id, stats in track_stats.items():

    # Calculate average bounding-box area
    average_area = (
        stats["total_area"] / stats["frames_seen"]
    )

    # Calculate average detection confidence
    average_confidence = (
        stats["total_confidence"] / stats["frames_seen"]
    )

    # Store the calculated statistics
    person_rankings.append({
        "track_id": track_id,
        "average_area": average_area,
        "average_confidence": average_confidence,
        "frames_seen": stats["frames_seen"]
    })


# --------------------------------
# Rank tracked people
# --------------------------------
# Ignore people detected in fewer than 30 frames
MIN_FRAMES = 30

person_rankings = [
    person for person in person_rankings
    if person["frames_seen"] >= MIN_FRAMES
]

if person_rankings:

    # Find the largest average area
    largest_average_area = max(
        person["average_area"]
        for person in person_rankings
    )

    # Calculate each person's score
    for person in person_rankings:

        area_score = (
            person["average_area"] / largest_average_area
        )

        confidence_score = person["average_confidence"]

        person["score"] = (
            0.7 * area_score +
            0.3 * confidence_score
        )

    # Sort people from highest score to lowest
    person_rankings.sort(
        key=lambda person: person["score"],
        reverse=True
    )

    # Display the ranking
    print("\n========== PERSON RANKINGS ==========")

    for rank, person in enumerate(person_rankings[:10], start=1):

        print(f"\nRank: {rank}")
        print("Tracking ID:", person["track_id"])
        print("Frames Seen:", person["frames_seen"])
        print("Average Area:", round(person["average_area"], 2))
        print(
            "Average Confidence:",
            round(person["average_confidence"], 2)
        )
        print("Score:", round(person["score"], 3))

    print("\n=====================================")


print("\nPerson tracking finished!")
