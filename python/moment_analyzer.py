
import json
from pathlib import Path
from ollama_client import ask_ollama


def analyze_moments(transcript_path):

    # Read the saved transcript
    with open(transcript_path, "r", encoding="utf-8") as file:
        segments = json.load(file)

    # Give every dialogue segment an index
    dialogue = ""

    for index, segment in enumerate(segments):
        dialogue += (
            f"[{index}] "
            f"({segment['start']:.2f}s - {segment['end']:.2f}s) "
            f"{segment['text']}\n"
        )

    # Prepare instructions for Llama 3
    prompt = f"""
You are an expert movie scene analyst.

Analyze the following movie dialogue and identify
up to 5 potentially interesting moments.

Look for:
- Funny moments
- Emotional moments
- Dramatic moments
- Motivational moments

IMPORTANT RULES:
1. Only select moments supported by the dialogue.
2. Use the exact segment numbers provided.
3. Start and end segment numbers must form a continuous range.
4. Do not invent dialogue or timestamps.
5. If there are no interesting moments, return an empty list.

Return ONLY valid JSON in this format:

{{
    "moments": [
        {{
            "start_segment": 0,
            "end_segment": 2,
            "category": "funny",
            "reason": "Why this moment is interesting",
            "hook": "A short title for this moment"
        }}
    ]
}}

MOVIE DIALOGUE:

{dialogue}
"""

    print("Llama 3 is analyzing the movie dialogue...")
    response = ask_ollama(prompt).strip()

    # Find where the JSON object starts and ends
    json_start = response.find("{")
    json_end = response.rfind("}")

    # Check whether JSON was found
    if json_start == -1 or json_end == -1:
        raise ValueError("Llama 3 did not return a JSON object.")

    # Extract and parse the JSON portion
    json_text = response[json_start:json_end + 1]
    result = json.loads(json_text)

    # Get the moments selected by the AI
    selected_moments = []

    for moment in result.get("moments", []):

        start_index = moment["start_segment"]
        end_index = moment["end_segment"]

        # Check whether the selected segment numbers are valid
        if not isinstance(start_index, int) or not isinstance(end_index, int):
            continue

        if start_index < 0 or end_index >= len(segments):
            continue

        if start_index > end_index:
            continue

        # Python retrieves the real timestamps
        selected_moments.append({
            "start": segments[start_index]["start"],
            "end": segments[end_index]["end"],
            "category": moment["category"],
            "reason": moment["reason"],
            "hook": moment["hook"]
        })

    return selected_moments
