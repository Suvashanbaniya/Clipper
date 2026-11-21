
import subprocess
from pathlib import Path


def extract_clip(video_path, start_time, end_time, output_path):

    # Create the output folder if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Calculate how long the clip should be
    duration = end_time - start_time

    # FFmpeg command to cut the video
    command = [
        "ffmpeg",
        "-ss", str(start_time),       # Starting timestamp
        "-i", video_path,             # Original video
        "-t", str(duration),          # Clip duration
        "-c:v", "libx264",            # Encode video
        "-c:a", "aac",                # Encode audio
        "-y",                         # Overwrite existing file
        output_path
    ]

    try:
        # Execute FFmpeg
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        return {
            "status": "success",
            "output_path": output_path,
            "duration": round(duration, 2),
            "message": "Video clip created successfully!"
        }

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Failed to create video clip. Check the input video."
        ) from error
