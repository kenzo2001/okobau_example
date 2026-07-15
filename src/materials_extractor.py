"""Architettura del file da separare prima della pipeline definitiva:
Comporta coesione medio alta ma medio accoppiamento tra i file. Un esempio di separazione potrebbe essere:
material_extractor-folder/
├── config.py
├── schemas.py
├── prompts.py
├── ollama_client.py
├── material_extractor.py
└── main.py
"""

import json, sys
from ollama import Client
from typing import List
from pydantic import BaseModel, Field, ValidationError
from shared import get_epds
from config import OLLAMA_HOST, LLM_MODEL, EMBED_MODEL


# Local ollama client
from ollama import Client

ollama_client = Client(
    host=OLLAMA_HOST
)
"""
# temporanei per test
print("OLLAMA_HOST:", OLLAMA_HOST, file=sys.stderr)
print("LLM_MODEL:", LLM_MODEL, file=sys.stderr)
print("EMBED_MODEL:", EMBED_MODEL, file=sys.stderr)
"""


class MaterialsResponse(BaseModel):
    materials: List[str] = Field(
        description = "Materials names extracted from the text"
    )


PROMPT = """
You are a deterministic construction-material extractor.

Extract only the names of the actual materials explicitly mentioned
in the input text.

Rules:
- Return the material itself, not the object or component made from it.
- Example: "Abstandhalter aus PVC" must return only "PVC".
- Keep the original language.
- Do not translate.
- Do not infer implicit or assumed materials.
- Exclude activities, operations, equipment, dimensions and performance values.
- Preserve grades and classes that identify materials, such as C25/30,
  B500B, CEM II/B or EPS.
- Do not return a composite product when its constituent materials are
  explicitly listed.
- Remove duplicates.
- Return only valid JSON matching the supplied schema.
"""

def cleaning_materials(materials: List[str]) -> List[str]:
    """Clean the list of materials by removing empty strings, whitespace and duplicates."""
    result = []
    duplicates = set()

    for value in materials:
        if not isinstance(value, str):
            continue
        name = " ".join(value.split())

        if not name:
            continue

        key = name.casefold()

        if key not in duplicates:
            duplicates.add(key)
            result.append(name)
        
    
    return result

def get_materials(text: str) -> List[str]:
    """Extract only the names of materials explicitly mentioned in the input text."""
    
    if not isinstance(text, str) or not text.strip():
        return []
    response = ollama_client.chat(
        model = LLM_MODEL,
        messages = [
            {
                "role": "system",
                "content": PROMPT
            },

            {
                "role": "user", 
                "content": f"Input text:\n{text.strip()}",
            },

        ],
        format = MaterialsResponse.model_json_schema(),
        think=False,
        stream = False,
        options={"temperature":0.0, "seed": 42, "num_predict": 150,},
    )

    content = response.message.content

    try:
        result = MaterialsResponse.model_validate_json(content)
    except ValidationError as e:
        raise RuntimeError(f"Qwen3 not returned valid JSON: {e}\n\nResponse content:\n{content}") from e
    
    return cleaning_materials(result.materials)

TRANSLATION_PROMPT = """
Translate each construction material name into Italian.

Rules:
- Preserve grades, classes, standards and abbreviations.
- Do not add explanations.
- Do not merge distinct materials.
- Keep the original name in the output.
- Return only valid JSON.
"""

def translate_materials_to_ita(materials: list[str]) -> list[dict]:
    if not materials:
        return[]
    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {
            "role": "system",
            "content": TRANSLATION_PROMPT
            },
            {
            "role": "user",
            "content": json.dumps({
                "materials": materials},
                ensure_ascii=False
                ),
            },
        ],
        format={
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_name": {
                                "type": "string"
                            },
                            "italian_name": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "original_name",
                            "italian_name",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        },

        think=False,
        stream=False,
        options={
            "temperature": 0,
            "seed": 42,
            "num_predict": 300,
        },
    )

    result = json.loads(response.message.content)

    return result ["translations"]

def main():
    

    try:
        # limit = 5 dovrebbe prendere i primi 5 record dal dataset
        data = get_epds(limit=5)
        epds = data.get("data", [])

        results = []
        total = len(epds)

        for index, epd in enumerate(epds, start=1):
            testo = epd.get("name", "")
            uuid = epd.get("uuid")

            print(
                f"[{index}/{total}] Elaborazione: {testo}",
                file=sys.stderr,
            )

            if not testo:
                results.append(
                    {
                        "uuid": uuid,
                        "source_text": "",
                        "materials": [],
                        "translations": [],
                        "status": "skipped",
                        "error": "Campo name mancante",
                    }
                )
                continue

            try:
                materials = get_materials(testo)
                translations = translate_materials_to_ita(materials)

                results.append(
                    {
                        "uuid": uuid,
                        "source_text": testo,
                        "materials": materials,
                        "translations": translations,
                        "status": "success",
                        "error": None,
                    }
                )

            except Exception as item_error:
                results.append(
                    {
                        "uuid": uuid,
                        "source_text": testo,
                        "materials": [],
                        "translations": [],
                        "status": "error",
                        "error": str(item_error),
                    }
                )

        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            )
        )

    except Exception as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()