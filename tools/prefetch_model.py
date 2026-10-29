#!/usr/bin/env python3
"""Prepare the exact locked model's Hugging Face cache, with explicit opt-in.

Default: print an offline plan, make no network requests, and download nothing.
With --download: fetch only essential configuration/tokenizer files and the
safetensors shards referenced by the pinned model index. Never load a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ESSENTIAL = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
}
REQUIRED = {"config.json", "tokenizer.json", "tokenizer_config.json"}
INDEX = "model.safetensors.index.json"
MARKER = "rsi-prefetch-request.json"
MANIFEST = "rsi-prefetch-manifest.json"
WEIGHT_NAME = re.compile(r"model(?:-\d{5}-of-\d{5})?\.safetensors\Z")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _model_lock(path):
    lock = json.loads(Path(path).read_text())
    model = lock.get("model", {})
    repo_id, revision = model.get("repo_id"), model.get("revision")
    if not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("The lock must contain an explicit owner/model repo_id")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The lock must contain a full 40-character commit revision")
    return {"repo_id": repo_id, "revision": revision}


def _weight_files(index, available):
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("The safetensors index must have a nonempty weight_map")
    values = list(mapping.values())
    if any(not isinstance(name, str) or not WEIGHT_NAME.fullmatch(name) for name in values):
        raise ValueError("Unsafe or non-safetensors filename in the pinned weight index")
    weights = sorted(set(values))
    missing = set(weights) - set(available)
    if missing:
        raise ValueError(f"Index references files absent from the pinned repository: {sorted(missing)}")
    return weights


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _verify_file(path, sibling):
    """Hash without loading a multi-GB weight shard into RAM."""
    path = Path(path)
    size = path.stat().st_size
    recorded_size = _get(sibling, "size")
    if recorded_size is not None and size != recorded_size:
        raise ValueError(f"Remote/local byte count differs: {path.name}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    digest = sha256.hexdigest()
    lfs = _get(sibling, "lfs")
    expected_lfs = _get(lfs, "sha256") if lfs is not None else None
    blob_id = _get(sibling, "blob_id")
    if expected_lfs:
        if digest != expected_lfs:
            raise ValueError(f"LFS SHA-256 mismatch: {path.name}")
    elif blob_id and re.fullmatch(r"[0-9a-f]{40}", blob_id):
        if git_blob.hexdigest() != blob_id:
            raise ValueError(f"Git blob hash mismatch: {path.name}")
    else:
        raise ValueError(f"Pinned repository did not supply a verifiable content hash: {path.name}")
    return {"bytes": size, "sha256": digest, "repository_blob_id": blob_id,
            "repository_lfs_sha256": expected_lfs}


def _write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(value))
    os.replace(temporary, path)


def download_locked_model(model, cache_dir):
    """Explicitly online operation. Called only after --download is supplied."""
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from importlib.metadata import version
    except ImportError as exc:
        raise RuntimeError("Install the project's gpu extra, which includes huggingface_hub via transformers") from exc
    if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}:
        raise RuntimeError("HF_HUB_OFFLINE is set; unset it only for this explicit prefetch step")
    cache_dir = Path(cache_dir)
    if cache_dir.is_symlink():
        raise ValueError("Use a real private cache directory, not a symlink")
    cache_dir = cache_dir.resolve()
    if cache_dir.exists():
        marker = cache_dir / MARKER
        if not marker.is_file() or json.loads(marker.read_text()) != model:
            raise ValueError("Existing cache is not marked for this exact model/revision; choose a new directory")
    else:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        _write_json(cache_dir / MARKER, model)
    os.chmod(cache_dir, 0o700)
    old_umask = os.umask(0o077)
    try:
        # This is a public, ungated model. Explicit token=False avoids sending
        # any cached user credential. No remote Python files are downloaded.
        info = HfApi(endpoint="https://huggingface.co", token=False).model_info(
            model["repo_id"], revision=model["revision"], files_metadata=True)
        if info.sha != model["revision"]:
            raise ValueError("Hub resolved a different revision than the preregistered lock")
        siblings = {item.rfilename: item for item in info.siblings}
        missing = REQUIRED - siblings.keys()
        if missing:
            raise ValueError(f"Required tokenizer/config files absent: {sorted(missing)}")
        kwargs = {"repo_id": model["repo_id"], "revision": model["revision"],
                  "cache_dir": str(cache_dir), "token": False,
                  "endpoint": "https://huggingface.co"}
        small_files = sorted(ESSENTIAL & siblings.keys())
        if INDEX in siblings:
            index_path = hf_hub_download(filename=INDEX, **kwargs)
            _verify_file(index_path, siblings[INDEX])
            if Path(index_path).stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Unexpectedly large safetensors index")
            weights = _weight_files(json.loads(Path(index_path).read_text()), siblings)
            selected = small_files + [INDEX] + weights
        elif "model.safetensors" in siblings:
            selected = small_files + ["model.safetensors"]
        else:
            raise ValueError("No recognized safetensors weights at the pinned revision")
        snapshot = Path(snapshot_download(
            allow_patterns=selected,
            ignore_patterns=["*.bin", "*.pkl", "*.pickle", "*.pt", "*.pth", "*.py"],
            max_workers=2, **kwargs)).resolve()
        if snapshot.name != model["revision"]:
            raise ValueError("Unexpected snapshot directory identity")
        files = {}
        for name in selected:
            files[name] = _verify_file(snapshot / name, siblings[name])
        for name in ("config.json", "tokenizer_config.json"):
            config = json.loads((snapshot / name).read_text())
            if config.get("auto_map"):
                raise ValueError("Pinned configuration requests remote code; this prefetch workflow refuses it")
        manifest = {
            "schema": "rsi-audit-model-prefetch-v1", **model,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "huggingface_hub_version": version("huggingface-hub"),
            "snapshot_relative_path": str(snapshot.relative_to(cache_dir.resolve())),
            "total_selected_bytes": sum(item["bytes"] for item in files.values()),
            "files": files, "weight_format": "safetensors", "model_loaded": False,
        }
        _write_json(cache_dir / MANIFEST, manifest)
        return manifest
    finally:
        os.umask(old_umask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evidence/baseline-lock.json")
    parser.add_argument("--cache-dir", "--output", dest="cache_dir", type=Path,
                        default=Path("private-model"),
                        help="HF cache root, passed unchanged to rsi_audit --model-cache")
    parser.add_argument("--download", action="store_true",
                        help="Explicitly allow network access and download the pinned model weights")
    args = parser.parse_args(argv)
    model = _model_lock(args.lock)
    if not args.download:
        print(json.dumps({"mode": "offline-plan", **model, "cache_dir": str(args.cache_dir),
                          "network_access": False, "weights_downloaded": False,
                          "small_file_allowlist": sorted(ESSENTIAL),
                          "weights": "Only safetensors filenames named by the pinned index (or model.safetensors)",
                          "next_step": "Add --download to explicitly fetch the exact revision"}, indent=2))
        return 0
    manifest = download_locked_model(model, args.cache_dir)
    print(json.dumps({"status": "verified", "repo_id": model["repo_id"],
                      "revision": model["revision"], "files": len(manifest["files"]),
                      "total_selected_bytes": manifest["total_selected_bytes"],
                      "manifest": str(args.cache_dir / MANIFEST), "model_loaded": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"prefetch failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
