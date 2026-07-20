"""Traduzione in italiano dei campi testuali del dataset EPD.

Logica:
- legge output/epd_attributes.csv;
- conserva tutte le colonne originali;
- aggiunge una colonna italiana per ogni campo traducibile;
- usa Ollama/Qwen tramite la configurazione esistente;
- memorizza le traduzioni in una cache JSON persistente;
- aggiorna progressivamente lo stesso CSV, così il lavoro può riprendere.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from ollama import Client
from pydantic import BaseModel, ValidationError

from config import LLM_MODEL, OLLAMA_HOST


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DEFAULT_CSV_FILE = PROJECT_FOLDER / "output" / "epd_attributes.csv"
DEFAULT_CACHE_FILE = PROJECT_FOLDER / "output" / "translation_cache.json"

# Campi contenenti nomi, descrizioni o classificazioni da tradurre.
# I campi numerici, UUID, anni, codici, unità e URL non vengono modificati.
TRANSLATE_COLUMNS = [
    "catalog_name",
    "name_de",
    "name_en",
    "classification_system",
    "classification",
    "general_comment_en",
    "geography",
]

# Nomi delle nuove colonne italiane. Le colonne originali restano intatte.
ITALIAN_COLUMN_NAMES = {
    "catalog_name": "nome_catalogo_it",
    "name_de": "nome_tedesco_it",
    "name_en": "nome_inglese_it",
    "classification_system": "sistema_classificazione_it",
    "classification": "classificazione_it",
    "general_comment_en": "commento_generale_it",
    "geography": "geografia_it",
}

TRANSLATION_PROMPT = """
You are a deterministic technical translator specialized in construction
materials, Environmental Product Declarations (EPD), and Life Cycle Assessment.

Translate the supplied German or English text into Italian.

Translate names, descriptions, classifications, categories, and ordinary
geographical names.

Preserve exactly and do not translate:
- UUIDs, URLs, email addresses, years, dates, and numeric values;
- registration numbers, dataset versions, product identifiers, and plant codes;
- standards and abbreviations such as EN, ISO, DIN, EPD, GWP, A1-A3, C25/30,
  XC4, XF1, XA1, B500B, CEM II/B, EPS, XPS, PIR, PUR;
- registered trademarks and commercial product names when translation would
  alter the identity of the product.

Rules:
- use professional Italian construction terminology;
- preserve every technical detail;
- do not summarize, explain, comment, or add information;
- return only valid JSON matching the supplied schema.
""".strip()


class TranslationResponse(BaseModel):
    italian_text: str


ollama_client = Client(host=OLLAMA_HOST)


def normalize_text(value: Any) -> str | None:
    """Restituisce una stringa pulita oppure None per valori vuoti/NaN."""
    if value is None or pd.isna(value):
        return None

    text = " ".join(str(value).split())
    return text or None


def load_translation_cache(cache_file: Path) -> dict[str, str]:
    """Carica la cache persistente delle traduzioni."""
    if not cache_file.exists():
        return {}

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Impossibile leggere la cache {cache_file}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            f"La cache {cache_file} deve contenere un oggetto JSON."
        )

    return {
        str(original): str(translation)
        for original, translation in data.items()
        if original is not None and translation is not None
    }


def save_translation_cache(
    cache: dict[str, str],
    cache_file: Path,
) -> None:
    """Salva la cache in modo atomico per ridurre il rischio di corruzione."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = cache_file.with_suffix(cache_file.suffix + ".tmp")

    temporary_file.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_file.replace(cache_file)


def translate_text_to_italian(text: str) -> str:
    """Traduce una singola stringa tramite Ollama e valida il JSON restituito."""
    response = ollama_client.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": TRANSLATION_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"text": text},
                    ensure_ascii=False,
                ),
            },
        ],
        format=TranslationResponse.model_json_schema(),
        think=False,
        stream=False,
        options={
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 1200,
        },
    )

    content = response.message.content

    try:
        parsed = TranslationResponse.model_validate_json(content)
    except ValidationError as error:
        raise RuntimeError(
            "Qwen3 non ha restituito un JSON valido.\n"
            f"Testo originale: {text}\n"
            f"Risposta: {content}"
        ) from error

    translation = normalize_text(parsed.italian_text)
    if not translation:
        raise RuntimeError(
            f"Ollama ha restituito una traduzione vuota per: {text}"
        )

    return translation


def translate_text_with_cache(
    text: str,
    cache: dict[str, str],
    cache_file: Path,
) -> tuple[str, bool]:
    """Restituisce la traduzione e indica se è stata chiamata Ollama."""
    normalized = normalize_text(text)
    if not normalized:
        return "", False

    cached = cache.get(normalized)
    if cached:
        return cached, False

    translation = translate_text_to_italian(normalized)
    cache[normalized] = translation
    save_translation_cache(cache, cache_file)
    return translation, True


def validate_csv_columns(dataframe: pd.DataFrame) -> list[str]:
    """Controlla le colonne e restituisce quelle traducibili presenti."""
    available = [
        column
        for column in TRANSLATE_COLUMNS
        if column in dataframe.columns
    ]

    if not available:
        raise ValueError(
            "Nel CSV non è presente nessuna delle colonne traducibili: "
            + ", ".join(TRANSLATE_COLUMNS)
        )

    missing = [
        column
        for column in TRANSLATE_COLUMNS
        if column not in dataframe.columns
    ]
    if missing:
        print(
            "Colonne opzionali non presenti e ignorate: "
            + ", ".join(missing),
            file=sys.stderr,
        )

    return available


def create_backup(csv_file: Path) -> Path:
    """Crea una copia di sicurezza una sola volta."""
    backup_file = csv_file.with_name(csv_file.stem + "_backup.csv")
    if not backup_file.exists():
        shutil.copy2(csv_file, backup_file)
        print(f"Backup creato: {backup_file}", file=sys.stderr)
    return backup_file


def save_dataframe(dataframe: pd.DataFrame, csv_file: Path) -> None:
    """Salva il CSV in modo atomico."""
    temporary_file = csv_file.with_suffix(csv_file.suffix + ".tmp")
    dataframe.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8-sig",
    )
    temporary_file.replace(csv_file)


def translate_epd_csv(
    csv_file: Path,
    cache_file: Path,
    limit: int | None = None,
    save_every: int = 20,
    overwrite_translations: bool = False,
    create_csv_backup: bool = True,
) -> None:
    """Traduce i campi testuali e aggiorna direttamente il CSV indicato."""
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_file}")

    if save_every < 1:
        raise ValueError("save_every deve essere almeno 1.")

    dataframe = pd.read_csv(
        csv_file,
        encoding="utf-8-sig",
        dtype=object,
        keep_default_na=True,
    )

    columns_to_translate = validate_csv_columns(dataframe)

    if create_csv_backup:
        create_backup(csv_file)

    # Non azzera le colonne già esistenti: consente la ripresa del lavoro.
    for source_column in columns_to_translate:
        italian_column = ITALIAN_COLUMN_NAMES[source_column]
        if italian_column not in dataframe.columns:
            dataframe[italian_column] = pd.NA

    cache = load_translation_cache(cache_file)

    # Conta i testi unici per mostrare l'effettivo risparmio di chiamate.
    unique_texts = {
        text
        for source_column in columns_to_translate
        for value in dataframe[source_column]
        if (text := normalize_text(value))
    }
    untranslated_unique = {
        text for text in unique_texts if text not in cache
    }

    print(f"Righe CSV: {len(dataframe)}", file=sys.stderr)
    print(f"Testi unici: {len(unique_texts)}", file=sys.stderr)
    print(
        f"Testi unici non ancora in cache: {len(untranslated_unique)}",
        file=sys.stderr,
    )

    new_ollama_calls = 0
    updated_cells = 0
    errors: list[dict[str, str]] = []

    for row_index, row in dataframe.iterrows():
        uuid = normalize_text(row.get("uuid")) or "UUID assente"

        for source_column in columns_to_translate:
            source_text = normalize_text(row.get(source_column))
            if not source_text:
                continue

            italian_column = ITALIAN_COLUMN_NAMES[source_column]
            current_translation = normalize_text(row.get(italian_column))

            if current_translation and not overwrite_translations:
                continue

            if limit is not None and new_ollama_calls >= limit:
                save_dataframe(dataframe, csv_file)
                print(
                    f"Raggiunto il limite di {limit} nuove chiamate Ollama.",
                    file=sys.stderr,
                )
                print(f"CSV aggiornato: {csv_file}", file=sys.stderr)
                return

            try:
                translation, called_ollama = translate_text_with_cache(
                    text=source_text,
                    cache=cache,
                    cache_file=cache_file,
                )
                dataframe.at[row_index, italian_column] = translation
                updated_cells += 1

                if called_ollama:
                    new_ollama_calls += 1
                    print(
                        f"[{row_index + 1}/{len(dataframe)}] "
                        f"{uuid} | {source_column} | nuova traduzione",
                        file=sys.stderr,
                    )

            except Exception as error:
                errors.append(
                    {
                        "uuid": uuid,
                        "column": source_column,
                        "text": source_text,
                        "error": str(error),
                    }
                )
                print(
                    f"Errore | {uuid} | {source_column}: {error}",
                    file=sys.stderr,
                )

            if updated_cells > 0 and updated_cells % save_every == 0:
                save_dataframe(dataframe, csv_file)

    save_dataframe(dataframe, csv_file)

    if errors:
        errors_file = csv_file.with_name(csv_file.stem + "_translation_errors.json")
        errors_file.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Errori salvati in: {errors_file}", file=sys.stderr)

    print(f"Celle aggiornate: {updated_cells}", file=sys.stderr)
    print(f"Nuove chiamate Ollama: {new_ollama_calls}", file=sys.stderr)
    print(f"Cache: {cache_file}", file=sys.stderr)
    print(f"CSV aggiornato: {csv_file}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traduce in italiano i campi testuali di epd_attributes.csv "
            "con Ollama e cache persistente."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_FILE,
        help=f"CSV da aggiornare (default: {DEFAULT_CSV_FILE})",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help=f"Cache JSON (default: {DEFAULT_CACHE_FILE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di nuove chiamate a Ollama, utile per i test.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Salva il CSV ogni N celle aggiornate (default: 20).",
    )
    parser.add_argument(
        "--overwrite-translations",
        action="store_true",
        help="Ricalcola anche le colonne italiane già valorizzate.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Non crea la copia epd_attributes_backup.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        translate_epd_csv(
            csv_file=args.csv,
            cache_file=args.cache,
            limit=args.limit,
            save_every=args.save_every,
            overwrite_translations=args.overwrite_translations,
            create_csv_backup=not args.no_backup,
        )
    except Exception as error:
        print(f"Errore: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()