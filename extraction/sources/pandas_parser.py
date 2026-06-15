import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def parse_excel(filepath: str) -> str:
    sheets = pd.read_excel(filepath, sheet_name=None, header=None)
    output: list[str] = []
    for sheet_name, frame in sheets.items():
        cleaned = frame.dropna(how="all", axis=0).dropna(how="all", axis=1)
        cleaned = cleaned.ffill(axis=0).ffill(axis=1).fillna("")
        if not cleaned.empty:
            output.append(f"--- Sheet: {sheet_name} ---")
            output.append(cleaned.to_csv(index=False, header=False))
    return "\n".join(output)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pandas_parser.py <filepath>", file=sys.stderr)
        sys.exit(1)
    try:
        print(parse_excel(sys.argv[1]))
    except Exception as exc:
        print(f"Error parsing {sys.argv[1]}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
