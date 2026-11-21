
import subprocess
from pathlib import Path


def extract_audio(video_path, audio_path):

    # Make sure the output folder exists
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract audio
    command = [
        "ffmpeg",
        "-i", video_path,       # Input video
        "-vn",                  # Ignore video
        "-ac", "1",             # Convert to mono audio
        "-ar", "16000",         # Set sample rate to 16 kHz
        "-c:a", "pcm_s16le",    # Save as WAV audio
        "-y",                   # Overwrite existing output
        audio_path
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
            "audio_path": audio_path,
            "message": "Audio extracted successfully!"
        }

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Audio extraction failed. Check the video file."
        ) from error
