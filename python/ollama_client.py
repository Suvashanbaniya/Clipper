
import json
import urllib.request


def ask_ollama(prompt):

    # Ollama API address
    url = "http://localhost:11434/api/generate"

    # Prepare the request data
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    # Convert Python data into JSON bytes
    json_data = json.dumps(data).encode("utf-8")

    # Create the HTTP request
    request = urllib.request.Request(
        url,
        data=json_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    # Send the request and receive the response
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Return the AI-generated response
    return result["response"]
