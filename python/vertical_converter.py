
import subprocess
from pathlib import Path


def convert_to_vertical(input_path, output_path):

    # Create the output folder if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command for full-screen vertical conversion
    command = [
        "ffmpeg",
        "-i", input_path,

        # Resize and crop the video to 9:16
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1",

        "-map", "0:v:0",
        "-map", "0:a?",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",

        "-c:a", "aac",
        "-b:a", "192k",

        "-movflags", "+faststart",
        "-y",
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
            "message": "Vertical video created successfully!"
        }

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Vertical video conversion failed."
        ) from error
