
import cv2


# Load the frontal face detection model
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Load the side-profile face detection model
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_profileface.xml"
)


def detect_faces(frame):

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces looking towards the camera
    frontal_faces = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    # Detect side-profile faces
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    # Flip the frame to detect profiles facing the opposite direction
    flipped_gray = cv2.flip(gray, 1)

    flipped_faces = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    # Convert flipped face coordinates back to original coordinates
    frame_width = gray.shape[1]

    for (x, y, width, height) in flipped_faces:

        original_x = frame_width - x - width

        profile_faces = list(profile_faces) + [
            (original_x, y, width, height)
        ]

    # Combine all detected faces
    all_faces = list(frontal_faces) + list(profile_faces)

    return all_faces
