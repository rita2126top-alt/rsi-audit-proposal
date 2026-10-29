"""Hash-chained, authenticated records owned by the trusted coordinator.

The key never enters a candidate sandbox. This protects against candidate writes,
not a malicious machine administrator who controls both the key and all files.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from .common import IntegrityError, canonical, digest


class Ledger:
    def __init__(self, directory: Path, *, create: bool = False):
        self.directory = Path(directory)
        self.path = self.directory / "ledger.jsonl"
        self.key_path = self.directory / "ledger.key"
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(os.urandom(32))
            self.path.touch(exist_ok=False, mode=0o600)
        if self.path.is_symlink() or self.key_path.is_symlink():
            raise IntegrityError("Ledger and key must not be symlinks")
        self.key = self.key_path.read_bytes()
        if len(self.key) != 32:
            raise IntegrityError("Invalid ledger key")
        self.records = self.verify()
        self.head = self.records[-1]["hash"] if self.records else "0" * 64

    def verify(self) -> list[dict]:
        records, previous = [], "0" * 64
        with self.path.open("rb") as stream:
            for number, raw in enumerate(stream):
                if not raw.endswith(b"\n"):
                    raise IntegrityError("Incomplete ledger record: refusing implicit truncation")
                try:
                    item = json.loads(raw)
                    payload = {"sequence": item["sequence"], "previous": item["previous"], "event": item["event"]}
                except (ValueError, KeyError, TypeError) as exc:
                    raise IntegrityError("Malformed ledger record") from exc
                if set(item) != {"sequence", "previous", "event", "hash", "hmac"}:
                    raise IntegrityError("Unexpected ledger envelope")
                if item["sequence"] != number or item["previous"] != previous:
                    raise IntegrityError("Broken ledger ordering")
                expected_hash = digest(payload)
                expected_mac = hmac.new(self.key, canonical(payload), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(item["hash"], expected_hash) or not hmac.compare_digest(item["hmac"], expected_mac):
                    raise IntegrityError("Ledger authentication failed")
                previous = item["hash"]
                records.append(item)
        return records

    def append(self, event: dict) -> dict:
        payload = {"sequence": len(self.records), "previous": self.head, "event": event}
        item = dict(payload, hash=digest(payload), hmac=hmac.new(self.key, canonical(payload), hashlib.sha256).hexdigest())
        with self.path.open("ab") as stream:
            stream.write(canonical(item) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.head = item["hash"]
        self.records.append(item)
        return item

    def assert_prefix(self, count: int, head: str) -> None:
        actual = "0" * 64 if count == 0 else self.records[count - 1]["hash"] if 0 < count <= len(self.records) else None
        if actual != head:
            raise IntegrityError("Frozen ledger prefix is missing or altered")

    def events(self, kind: str | None = None) -> list[dict]:
        return [x["event"] for x in self.records if kind is None or x["event"].get("kind") == kind]
