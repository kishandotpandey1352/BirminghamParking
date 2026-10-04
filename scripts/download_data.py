"""Download the official UCI archive; pass --local-csv to use an existing copy."""
import argparse
import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://archive.ics.uci.edu/static/public/482/parking+birmingham.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("parking_birmingham/dataset.csv"))
    parser.add_argument("--local-csv", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.local_csv:
        args.output.write_bytes(args.local_csv.read_bytes())
    else:
        payload = urlopen(URL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csvs = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csvs:
                raise RuntimeError("The official archive contained no CSV")
            args.output.write_bytes(archive.read(csvs[0]))
    print(f"Wrote {args.output}; sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
