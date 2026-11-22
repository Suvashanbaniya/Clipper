
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


# --------------------------------
# 1. Locate project folders
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

video_path = (
    PROJECT_ROOT / "input" / "movies" / "test_movie.mp4"
)

output_folder = PROJECT_ROOT / "temp"
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "smart_crop_tracking_preview.mp4"


# --------------------------------
# 2. Select GPU or CPU
# --------------------------------

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", "GPU" if device == 0 else "CPU")

if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------
# 3. Load YOLO model
# --------------------------------

print("\nLoading YOLO model...")

model = YOLO("yolo11n.pt")


# --------------------------------
# 4. Open the video
# --------------------------------

video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise RuntimeError("Could not open the movie.")

# Get video information
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo information:")
print("Width:", frame_width)
print("Height:", frame_height)
print("FPS:", fps)
print("Total frames:", total_frames)


# --------------------------------
# 5. Calculate vertical crop size
# --------------------------------

# Use the largest 9:16 crop that fits inside the source video
crop_height = frame_height
crop_width = int(crop_height * 9 / 16)

if crop_width > frame_width:
    crop_width = frame_width
    crop_height = int(crop_width * 16 / 9)

print("\nVertical crop dimensions:")
print("Crop width:", crop_width)
print("Crop height:", crop_height)


# --------------------------------
# 6. Prepare output video
# --------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (1080, 1920)
)

if not writer.isOpened():
    video.release()
    raise RuntimeError("Could not create the output video.")


# --------------------------------
# 7. Tracking and selection settings
# --------------------------------

# Store information about each tracked person
track_stats = {}

# Currently selected person
selected_track_id = None

# Current horizontal crop centre
smoothed_center_x = frame_width / 2

# Controls how quickly the crop follows the person
SMOOTHING_FACTOR = 0.18

# Minimum number of frames needed for full persistence score
PERSISTENCE_FRAMES = 60

# A new person must score this much higher to replace
# the currently selected person
SWITCH_THRESHOLD = 1.25

frame_number = 0


# --------------------------------
# 8. Process every video frame
# --------------------------------

print("\nStarting person tracking and smart cropping...")

try:

    while True:

        # Read the next video frame
        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        # Detect and track people
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

        # Store information about people in this frame
        people = []

        if result.boxes is not None and result.boxes.id is not None:

            for box in result.boxes:

                # Get tracking ID
                track_id = int(box.id[0].item())

                # Get bounding-box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Keep coordinates inside the frame
                x1 = max(0, min(x1, frame_width))
                y1 = max(0, min(y1, frame_height))
                x2 = max(0, min(x2, frame_width))
                y2 = max(0, min(y2, frame_height))

                width = x2 - x1
                height = y2 - y1

                if width <= 0 or height <= 0:
                    continue

                # Calculate person area
                area = width * height

                # Get detection confidence
                confidence = float(box.conf[0])

                # Extract the person's image region
                person_region = frame[y1:y2, x1:x2]

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

                # Calculate the person's horizontal centre
                center_x = (x1 + x2) / 2

                # Create statistics for a new tracking ID
                if track_id not in track_stats:

                    track_stats[track_id] = {
                        "frames_seen": 0,
                        "total_area": 0,
                        "total_sharpness": 0,
                        "total_confidence": 0
                    }

                # Update tracking statistics
                stats = track_stats[track_id]

                stats["frames_seen"] += 1
                stats["total_area"] += area
                stats["total_sharpness"] += sharpness
                stats["total_confidence"] += confidence

                # Store current detection
                people.append({
                    "track_id": track_id,
                    "box": (x1, y1, x2, y2),
                    "area": area,
                    "sharpness": sharpness,
                    "confidence": confidence,
                    "center_x": center_x
                })

        # --------------------------------
        # 9. Estimate the most prominent person
        # --------------------------------

        if people:

            max_area = max(
                person["area"] for person in people
            )

            max_sharpness = max(
                person["sharpness"] for person in people
            )

            for person in people:

                stats = track_stats[person["track_id"]]

                # Normalize the person's size
                area_score = person["area"] / max_area

                # Normalize image sharpness
                if max_sharpness > 0:
                    sharpness_score = (
                        person["sharpness"] / max_sharpness
                    )
                else:
                    sharpness_score = 0

                # Measure how consistently this person is tracked
                persistence_score = min(
                    stats["frames_seen"] / PERSISTENCE_FRAMES,
                    1.0
                )

                # Combine the four measurements
                person["score"] = (
                    0.40 * area_score +
                    0.25 * sharpness_score +
                    0.15 * person["confidence"] +
                    0.20 * persistence_score
                )

            # Find the highest-scoring person
            best_person = max(
                people,
                key=lambda person: person["score"]
            )

            # Check whether the previously selected person
            # is still visible in the current frame
            current_person = next(
                (
                    person for person in people
                    if person["track_id"] == selected_track_id
                ),
                None
            )

            # Select or switch the tracked person
            if current_person is None:

                selected_person = best_person

            elif (
                best_person["track_id"] != selected_track_id
                and best_person["score"]
                > current_person["score"] * SWITCH_THRESHOLD
            ):

                selected_person = best_person

            else:

                selected_person = current_person

            selected_track_id = selected_person["track_id"]

            # Get the selected person's centre
            target_center_x = selected_person["center_x"]

        else:

            # If nobody is detected, keep the previous crop centre
            target_center_x = smoothed_center_x

        # --------------------------------
        # 10. Smooth crop movement
        # --------------------------------

        smoothed_center_x = (
            SMOOTHING_FACTOR * target_center_x
            + (1 - SMOOTHING_FACTOR) * smoothed_center_x
        )

        # Calculate crop's left boundary
        crop_x = int(
            smoothed_center_x - crop_width / 2
        )

        # Keep crop inside the original frame
        crop_x = max(
            0,
            min(crop_x, frame_width - crop_width)
        )

        crop_x2 = crop_x + crop_width

        # Extract the vertical crop
        cropped_frame = frame[
            0:crop_height,
            crop_x:crop_x2
        ]

        # Resize to 1080 x 1920
        vertical_frame = cv2.resize(
            cropped_frame,
            (1080, 1920),
            interpolation=cv2.INTER_LINEAR
        )

        # Write the frame into the output video
        writer.write(vertical_frame)

        # Print progress every 100 frames
        if frame_number % 100 == 0:

            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

            print("Selected tracking ID:", selected_track_id)
            print("Crop X:", crop_x)

finally:

    # Release video resources
    video.release()
    writer.release()


# --------------------------------
# 11. Display final results
# --------------------------------

print("\nSmart cropping completed!")

print("Frames processed:", frame_number)
print("Unique tracking IDs:", len(track_stats))
print("Selected tracking ID:", selected_track_id)

print("\nSaved vertical preview:")
print(output_path)
