
import whisper
import torch


def transcribe_audio(audio_path):

    # Check whether an NVIDIA GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Whisper on: {device}")

    # Load the Whisper Base model
    model = whisper.load_model("base", device=device)

    print("Transcribing audio...")

    # Convert speech into text with timestamps
    result = model.transcribe(
        audio_path,
        fp16=(device == "cuda")
    )

    # Extract individual speech segments
    segments = []

    for segment in result["segments"]:
        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    return segments
