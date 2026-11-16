"""Record the paths, provenance, byte counts and hashes of acquired artifacts."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PUBLISHER_HASHES = {
    "model": "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
    "cpu": "33be28730fb2d6e16782870c6411b055b52fb4ffb025a28b52f0e74a0d714851",
    "cuda": "1f39036adc9529a0500ad7cb9c234de634219039035e83ac2b22e0a8fb784e04",
    "cudart": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
}


def inventory() -> None:
    sources = json.loads(Path("sources.json").read_text(encoding="utf-8"))
    sections = {"data": [], "model": [], "software": []}
    for key, name, url in sources:
        path = Path(name)
        entry = {"name": key, "path": name, "source": url, "status": "pending"}
        if path.is_file():
            stat = path.stat()
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            entry.update(
                bytes=stat.st_size, sha256=digest, status="acquired",
                acquired_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            )
            if key in PUBLISHER_HASHES:
                if digest != PUBLISHER_HASHES[key]:
                    raise ValueError(f"Publisher SHA-256 mismatch: {name}")
                entry["publisher_sha256_verified"] = True
        if name.startswith("data/"):
            entry["license"] = "CC BY-SA 4.0" if key.startswith("hot") else "CC BY 4.0"
            sections["data"].append(entry)
        elif name.startswith("models/"):
            entry["license"] = "Qwen Research License"
            sections["model"].append(entry)
        else:
            sections["software"].append(entry)
    destination = Path("downloads.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(sections, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    inventory()
