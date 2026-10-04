import argparse
from pathlib import Path

from parkpulse.model import train_model

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--version", default="real-2016-01")
args = parser.parse_args()
artifact_dir = ROOT / "artifacts" / "versions" / args.version if args.version != "real-2016-01" else ROOT / "artifacts"
manifest = train_model(ROOT / "parking_birmingham" / "dataset.csv", artifact_dir, version=args.version)
print(manifest["metrics"])
print(f"Wrote {artifact_dir / 'manifest.json'}")
