"""Extract plain text from Word documents (.docx, .doc) for solicitation extraction."""

from __future__ import annotations

import sys


def extract_word(filepath: str) -> str:
    import mammoth

    with open(filepath, "rb") as handle:
        result = mammoth.extract_raw_text(handle)
    text = result.value or ""
    if result.messages:
        for message in result.messages:
            print(message, file=sys.stderr)
    return text


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python word_parser.py <filepath>", file=sys.stderr)
        sys.exit(1)
    try:
        text = extract_word(sys.argv[1])
        if text.strip():
            print(text)
    except Exception as exc:
        print(f"Error parsing {sys.argv[1]}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
