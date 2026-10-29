#!/usr/bin/env python3
"""Download only the two explicitly pinned public benchmark archives.

Creates a new private directory, verifies archive and decoded JSONL hashes,
and refuses overwrite. Model weights are managed separately by the operator.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi_audit.data import DataError, load_evalplus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evidence/baseline-lock.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New private asset directory; existing directories are refused")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    records = {r["name"]: r for r in lock["datasets"]}
    selected = [("HumanEval+", "HumanEvalPlus.jsonl.gz"), ("MBPP+", "MbppPlus.jsonl.gz")]
    for name, _ in selected:
        record = records[name]
        parsed = urllib.parse.urlparse(record["asset_url"])
        if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith("/evalplus/") or "/releases/download/" not in parsed.path:
            raise DataError("Only the explicitly pinned EvalPlus GitHub release URLs are accepted")
        if type(record["gzip_bytes"]) is not int or not 0 < record["gzip_bytes"] < 64 * 1024 * 1024:
            raise DataError("Invalid locked asset byte count")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    for name, filename in selected:
        record = records[name]
        path = args.output / filename
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        request = urllib.request.Request(record["asset_url"], headers={"User-Agent": "RSI-Audit-pinned-asset-downloader/1"})
        total = 0
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > record["gzip_bytes"]:
                        raise DataError(f"Download exceeds locked byte count: {name}")
                    output.write(chunk)
        if total != record["gzip_bytes"]:
            raise DataError(f"Download byte count mismatch: {name}")
    tasks = load_evalplus(args.output / "HumanEvalPlus.jsonl.gz",
                          args.output / "MbppPlus.jsonl.gz", args.lock)
    print(json.dumps({"verified_tasks": len(tasks), "archives": [filename for _, filename in selected],
                      "status": "archive and decoded payload hashes verified"}))


if __name__ == "__main__":
    main()
