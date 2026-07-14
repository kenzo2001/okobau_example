import os, requests

OLLAMA = os.environ.get("OLLAMA_API_KEY")
#qui devi installare ollama, lo stronzo gira in locale su una porta e cercala e mettila in env
LLM_MODEL = os.getenv("LLM_MODEL", "llama2")  # Default model is llama2, cerca un modello adatto, magari deepseek
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")  # Default embedding model is text-embedding-3-large
 

def get_Answer(prompt: str, model: str = LLM_MODEL):
    body = {
        "model": model, 
        "prompt": prompt + "\n\n### traduci il tedesco",
    }
    r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

