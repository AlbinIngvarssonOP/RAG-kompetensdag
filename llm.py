"""Local Ollama client configuration and a single native chat call."""

from ollama import Client


OLLAMA_MODEL = "qwen3:4b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 180


def create_client():
    client = Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    return client, OLLAMA_MODEL


def send_message(client, model, messages):
    response = client.chat(model=model, messages=messages, stream=False)
    answer = response.message.content
    if not answer or not answer.strip():
        raise ValueError("The model returned no text.")
    return answer
