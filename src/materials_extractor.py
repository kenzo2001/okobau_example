"""Struttura per traduzione da tedesco/inglese in italiano
"""

import json, sys
import pandas as pd
import pathlib as Path
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

TRANSLATE_COLUMNS = [
    "catalog_name",
    "name_de",
    "name_en",
    "classification_system",
    "classification",
    "general_comment_en"
]

# sequenza per risparmio token, elimina duplicati
df = pd.read_csv("output/epd_attributes.csv", encoding="utf-8-sig")

unique_texts = set()

for column in TRANSLATE_COLUMNS:
    for value in df[column].dropna():

        text = str(value).strip()

        if text: unique_texts.add(text)

print("Testi unici: ", len(unique_texts))


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
You are a technical translator specialized in construction materials,
EPD and Life Cycle Assessment.

Translate the provided construction dataset fields into Italian.

Translate:
- product names
- classifications
- descriptions
- categories
- geographical descriptions

Do NOT translate:
- UUID
- numbers
- years
- URLs
- registration codes
- dataset versions
- GWP values
- technical codes

Preserve exactly:
- C25/30
- XC4 XF1
- EN 15804
- ISO 14025
- DIN
- EPS
- XPS
- PIR
- PUR

Rules:
- Use Italian construction terminology.
- Do not summarize.
- Do not explain.
- Do not remove technical information.
- Return only JSON.
"""



def translate_epd_fields(
    fields: dict
) -> dict:


    response = ollama_client.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role":"system",
                "content":TRANSLATION_PROMPT
            },
            {
                "role":"user",
                "content":json.dumps(
                    fields,
                    ensure_ascii=False
                )
            }
        ],

        format={
            "type":"object",
            "additionalProperties":{
                "type":"string"
            }
        },

        think=False,
        stream=False,

        options={
            "temperature":0,
            "seed":42,
            "num_predict":500
        }
    )


    return json.loads(
        response.message.content
    )



#main di testing su 5 record
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



if __name__=="__main__":
    main()

