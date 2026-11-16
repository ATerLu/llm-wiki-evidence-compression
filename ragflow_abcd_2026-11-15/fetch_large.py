"""Fetch a pinned public artifact in validated HTTP ranges via the Windows proxy."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import shutil
import subprocess


def request(url: str, proxy: str, span: str, output: Path) -> str:
    result = subprocess.run([
        "curl.exe", "--proxy", proxy, "-fLsS", "--connect-timeout", "15", "--max-time", "180",
        "--retry", "4", "--retry-all-errors", "--retry-delay", "1",
        "-r", span, "-D", "-", "-o", str(output), url,
    ], check=True, capture_output=True, text=True)
    return result.stdout


def fetch(url: str, output: Path, proxy: str, expected: str | None) -> None:
    scratch = output.with_name(output.name + ".ranges")
    scratch.mkdir(exist_ok=True)
    separator = "&" if "?" in url else "?"
    probe = request(url + separator + "range_probe=1", proxy, "0-0", scratch / "probe")
    match = re.search(r"^content-range: bytes 0-0/(\d+)", probe, re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError("Server did not advertise a valid ranged response")
    size = int(match[1])
    chunk_size = 32 * 1024 * 1024

    def fetch_part(start: int) -> Path:
        end = min(size, start + chunk_size) - 1
        part = scratch / str(start)
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        temporary = part.with_suffix(".partial")
        headers = request(url + separator + f"range_start={start}", proxy, f"{start}-{end}", temporary)
        wanted = f"content-range: bytes {start}-{end}/{size}"
        if wanted not in headers.lower().splitlines() or temporary.stat().st_size != end - start + 1:
            raise ValueError(f"Invalid response for range {start}-{end}")
        temporary.replace(part)
        print(f"{output.name}: verified {start}-{end}/{size}", flush=True)
        return part

    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(fetch_part, range(0, size, chunk_size)))
    temporary = scratch / "combined"
    with temporary.open("wb") as file:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, file, 1024 * 1024)
    with temporary.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if expected and digest != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {output.name}: {digest}")
    temporary.replace(output)
    for part in [*parts, scratch / "probe"]:
        part.unlink()
    scratch.rmdir()
    print(f"{output}: {size} bytes, SHA-256 {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    try:
        fetch(args.url, args.output, args.proxy, args.sha256)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr.strip()) from error
