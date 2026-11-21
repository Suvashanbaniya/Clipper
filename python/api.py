
from flask import Flask, jsonify, request

from pathlib import Path
from video_analyzer import inspect_video

# Create our Flask application
app = Flask(__name__)


# Check whether the API is running
@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "success",
        "message": "AI Movie Clipper API is running!",
        "service": "Movie Clipper"
    })


# Receive video metadata from n8n
@app.route("/analyze", methods=["POST"])
def analyze_video():

    # Read the JSON data sent by n8n
    data = request.get_json()

    # Check whether data was received
    if not data:
        return jsonify({
            "status": "error",
            "message": "No video data received"
        }), 400

    # Check for required information
    if not data.get("video_id") or not data.get("title"):
        return jsonify({
            "status": "error",
            "message": "Video ID and title are required"
        }), 400

    # Return the received video information
    return jsonify({
        "status": "success",
        "message": "Video metadata received successfully!",
        "video_id": data["video_id"],
        "title": data["title"],
        "channel": data.get("channel", "Unknown")
    })

# Inspect a video stored in input/movies
@app.route("/inspect", methods=["POST"])
def inspect_movie():

    # Get the filename sent by n8n
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")

    # Check whether a filename was provided
    if not filename:
        return jsonify({
            "status": "error",
            "message": "Filename is required"
        }), 400

    # Accept only a filename, not a folder path
    if Path(filename).name != filename:
        return jsonify({
            "status": "error",
            "message": "Send only a filename, not a path"
        }), 400

    # Locate the video inside our project's input/movies folder
    project_root = Path(__file__).resolve().parent.parent
    movies_folder = project_root / "input" / "movies"
    video_path = movies_folder / filename

    # Check whether the video exists
    if not video_path.is_file():
        return jsonify({
            "status": "error",
            "message": "Video file not found in input/movies"
        }), 404

    try:
        # Use our video analyzer to inspect the video
        video_info = inspect_video(str(video_path))

        return jsonify({
            "status": "success",
            "message": "Video inspected successfully",
            "filename": filename,
            "video_info": video_info
        })

    except (RuntimeError, ValueError) as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400
# Start the server
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )
