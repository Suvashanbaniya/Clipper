
import subprocess
import json


def inspect_video(video_path):

    # Run FFprobe to collect video information
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        video_path
    ]

    try:
        # Execute the FFprobe command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        # Convert FFprobe JSON output into Python data
        video_data = json.loads(result.stdout)

        # Find the video and audio streams
        video_stream = next(
            (s for s in video_data.get("streams", [])
             if s["codec_type"] == "video"),
            None
        )

        audio_stream = next(
            (s for s in video_data.get("streams", [])
             if s["codec_type"] == "audio"),
            None
        )

        if video_stream is None:
            raise ValueError("No video stream found")

        # Prepare useful video information
        return {
            "duration": float(video_data["format"]["duration"]),
            "size_mb": round(
                int(video_data["format"]["size"]) / (1024 * 1024), 2
            ),
            "width": video_stream["width"],
            "height": video_stream["height"],
            "frame_rate": video_stream["r_frame_rate"],
            "video_codec": video_stream["codec_name"],
            "audio_available": audio_stream is not None,
            "audio_codec": (
                audio_stream["codec_name"] if audio_stream else None
            )
        }

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "FFprobe could not inspect the video. Check the file."
        ) from error
