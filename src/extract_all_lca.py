import json, sys, time
from pathlib import Path

from shared import get_all_epds, get_lca_by_uuid


OUTPUT_FOLDER = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_FOLDER / "lca_products.json"

## --------- PRODOTTI PER TEST --------
    # Imposta a None se vogliamo farlo su tutto
MAX_PRODUCTS = 5


def extract_all_lca(max_products: int | None = None) -> list[dict]:
    """Extract present products in database named "ÖKOBAUDAT"
     and try to extract all LCA"""
    
    epds = get_all_epds()

    if max_products is not None:
        epds = epds[:max_products]

    results = []

    total = len(epds)

    for index, epd in enumerate(epds, start=1):
        uuid = epd.get("uuid")
        name = epd.get("name")


        print(f"[{index}/{total}] Elaborazione: {name}")

        if not uuid:
            results.append(
                {
                    "uuid": None,
                    "name": name,
                    "status": "error",
                    "error": "UUID not found",
                    "lca": []
                }
            )
            continue
        try:
            lca_data = get_lca_by_uuid(
                uuid=uuid,
                name=name
            )
            results.append(
                {
                    **lca_data,
                    "status": "success",
                    "error": None
                }
                )
        except Exception as error:
            print(f"Errore per {name}: {error}", 
            file = sys.stderr
            )
        time.sleep(0.2)

    return results


def save_results(results: list[dict]) -> None:
    """Save all results in a JSON file"""

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"
    )

    print(f"\nResults saved in: {OUTPUT_FILE}")

def main():
    print("Avvio script")

    results = extract_all_lca(
        max_products=MAX_PRODUCTS
    )

    print("Estrazione completata")
    save_results(results)

    print("Salvataggio completato")

    successful = sum(
        1
        for item in results if item["status"] == "success"
    )

    errors = len(results) - successful

    #resoconto
    print(f"Elaborated products: {len(results)}")
    print(f"Extracted LCA: {successful}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()

