import argparse
import json

from shared import find_epds_by_name, get_lca_by_epd_match


def main():
    parser = argparse.ArgumentParser(
        description="Find an EPD by object name and extract its LCA results."
    )
    parser.add_argument("object_name", help="Object name to search in the EPD list.")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require an exact match instead of a substring match.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matching EPD metadata without downloading the full LCA dataset.",
    )
    args = parser.parse_args()

    matches = find_epds_by_name(args.object_name, exact=args.exact)
    if args.list_only or len(matches) != 1:
        print(json.dumps(matches, indent=2, ensure_ascii=False))
        return

    result = get_lca_by_epd_match(matches[0], object_name=args.object_name)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()