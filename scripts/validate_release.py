import json
import sys
from pathlib import Path


def main() -> None:
    manifest_path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"model_version", "feature_schema", "dataset_sha256", "metrics", "split_boundaries"}
    missing = required - manifest.keys()
    if missing or manifest.get("test_only", True):
        raise SystemExit(f"Release rejected: missing={sorted(missing)} test_only={manifest.get('test_only')}")
    print(f"Release valid: {manifest['model_version']}")


if __name__ == "__main__":
    main()