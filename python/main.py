
from pathlib import Path
from moment_analyzer import analyze_moments
from clip_extractor import extract_clip
from vertical_converter import convert_to_vertical

# Location of our original video
video_path = "input/movies/test_movie.mp4"

# Location of our transcript
transcript_path = "temp/transcription.json"

# Folder where generated clips will be saved
output_folder = Path("output/clips")


try:
    # Ask Llama 3 to identify interesting moments
    moments = analyze_moments(transcript_path)

    print("\n--- AI SELECTED MOVIE MOMENTS ---")

    if not moments:
        print("No interesting moments found.")

    # Generate a video clip for every selected moment
    for index, moment in enumerate(moments, start=1):

        print(f"\nProcessing Moment {index}")

        print("Category:", moment["category"])
        print("Hook:", moment["hook"])

        print(
            f"Timestamp: {moment['start']:.2f}s - "
            f"{moment['end']:.2f}s"
        )

        # Create a unique filename for each clip
        output_path = output_folder / f"moment_{index}.mp4"

        # Cut the selected moment from the original video
        result = extract_clip(
            video_path,
            moment["start"],
            moment["end"],
            str(output_path)
        )

        print(result["message"])
        print("Saved to:", result["output_path"])

        # Create the vertical video filename
        vertical_path = Path("output/final") / f"moment_{index}_vertical.mp4"

        # Convert the extracted clip into vertical format
        vertical_result = convert_to_vertical(
            str(output_path),
            str(vertical_path)
        )

        print(vertical_result["message"])
        print("Vertical video saved to:", vertical_result["output_path"])
    print("\nAll clips have been processed!")

except Exception as exc:
    print("Error:", exc)
