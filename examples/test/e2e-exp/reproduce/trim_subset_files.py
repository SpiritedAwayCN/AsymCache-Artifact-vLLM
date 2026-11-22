#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path


def trim_file(path: Path, keep_lines: int, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] keep first {keep_lines} lines: {path}")
        return

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out_f, open(path, "r", encoding="utf-8") as in_f:
            out_f.writelines(islice(in_f, keep_lines))
        shutil.copystat(path, tmp_path)
        os.replace(tmp_path, path)
        print(f"[OK] trimmed: {path}")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim all .subset files under reproduce to the first N lines.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--lines", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.lines < 0:
        raise ValueError("--lines must be non-negative")

    files = sorted(p for p in args.root.rglob("*.subset") if p.is_file())
    print(f"Found {len(files)} .subset files under {args.root}")

    for path in files:
        trim_file(path, args.lines, args.dry_run)


if __name__ == "__main__":
    main()
