"""Explicit commands for preparation, search, freeze, audit and analysis."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .analysis import analyze_run, write_report
from .common import IntegrityError, atomic_json, file_digest, read_json
from .data import load_evalplus, partition_tasks, pilot_bundle_from_dev, verify_commitment, write_bundle
from .engine import Engine, profile, verify_run
from .solver import RealAdapter, SyntheticAdapter


def doctor() -> dict:
    packages = {}
    for package in ("numpy", "torch", "transformers", "huggingface-hub"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    gpu = None
    if importlib.util.find_spec("torch"):
        import torch
        gpu = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "docker_cli": shutil.which("docker"), "gpu": gpu,
            "note": "This command detects dependencies; it does not certify the daemon, worker isolation or model correctness."}


def prepare(args) -> dict:
    from .execution import DockerExecutor
    from .scoring import score_outputs
    config = profile(read_json(args.config), "formal")
    lock = read_json(args.lock)
    if config["model"]["id"] != lock["model"]["repo_id"] or config["model"]["revision"] != lock["model"]["revision"]:
        raise ValueError("Config and evidence model revisions differ")
    if args.out.exists():
        raise FileExistsError("Preparation refuses to overwrite an existing bundle")
    tasks = load_evalplus(args.assets / "HumanEvalPlus.jsonl.gz", args.assets / "MbppPlus.jsonl.gz", args.lock)
    limits = config["engine"]["limits"]
    executor = DockerExecutor(args.image, **limits)
    started = time.monotonic()
    execution_seconds = 0.0
    for index, task in enumerate(tasks, 1):
        outcome = executor.execute(task["canonical_code"], task["entry_point"], task["inputs"])
        execution_seconds += outcome["wall_seconds"]
        if outcome["status"] != "ok":
            raise RuntimeError(f"Canonical oracle failed for {task['task_id']}: {outcome['status']}; no task was discarded")
        task["expected"] = outcome["outputs"]
        check = score_outputs(task, outcome)
        if not check["passed"]:
            raise RuntimeError(f"Canonical oracle verification failed for {task['task_id']}: {check['status']}")
        if index % 25 == 0 or index == len(tasks):
            print(f"Canonical preparation: {index}/{len(tasks)} tasks", file=sys.stderr, flush=True)
    overrides = read_json(args.family_map) if args.family_map else None
    partitions = partition_tasks(tasks, config["data"]["target_dev"], config["search"]["rounds"],
                                 config["gate"]["panel_tasks"], config["data"]["minimum_audit"], args.seed,
                                 family_overrides=overrides)
    metadata = {"purpose": "formal-study", "oracle_precomputed": True, "oracle_executor_identity": executor.identity,
                "oracle_execution_limits": limits, "seed": args.seed, "lock_sha256": file_digest(args.lock),
                "family_overrides_sha256": file_digest(args.family_map) if args.family_map else None,
                "preparation": {"wall_seconds": time.monotonic() - started, "execution_seconds": execution_seconds,
                                "gpu_hours": 0.0, "canonical_tasks": len(tasks), "cpu_seconds": None},
                "note": "Canonical programs execute only in Docker; this trusted offline preparation never supplies answers to the model."}
    manifest = write_bundle(partitions, args.out, metadata)
    verify_commitment(args.out)
    return {"bundle": str(args.out), "manifest": manifest,
            "next_step": "Archive public_manifest.json with an external timestamp before formal search; retain private files separately."}


def make_pilot(args) -> dict:
    parent_metadata = read_json(args.bundle / "metadata.private.json")
    manifest = read_json(args.bundle / "public_manifest.json")
    if file_digest(args.bundle / "metadata.private.json") != manifest["files"]["metadata.private.json"]:
        raise IntegrityError("Parent metadata differs from commitment")
    if not parent_metadata.get("oracle_precomputed") or parent_metadata.get("purpose") != "formal-study":
        raise ValueError("Pilot requires a prepared formal parent bundle")
    result = pilot_bundle_from_dev(args.bundle, args.out, dev_count=16, panel_count=2, panel_size=8,
                                   audit_min=8, audit_count=8, seed=args.seed, metadata=parent_metadata)
    return {"bundle": str(args.out), "manifest": result, "scope": "40 tasks selected exclusively from parent development families"}


def real_adapter(args, config):
    from .backend import TransformersBackend
    from .execution import DockerExecutor
    started = time.monotonic()
    executor = DockerExecutor(args.image, **config["engine"]["limits"])
    backend = TransformersBackend(config["model"]["id"], config["model"]["revision"],
                                  cache_dir=str(args.model_cache), device=args.device)
    return RealAdapter(backend, executor, config), time.monotonic() - started


def search(args) -> dict:
    config = profile(read_json(args.config), args.mode)
    if args.run.exists():
        verify_run(args.run)
        if read_json(args.run / "config.json") != config:
            raise IntegrityError("Resume configuration differs from initialized run")
        previous = read_json(args.run / "state.json")
        if previous["phase"] == "searched":
            return {"run": str(args.run), "phase": "searched", "mode": previous["mode"],
                    "runs_completed": len(previous["runs"]), "model_loaded": False}
        if previous["phase"] != "searching":
            raise IntegrityError("Cannot start search after freeze")
    adapter, startup = real_adapter(args, config)
    engine = Engine(args.run, adapter=adapter, bundle_dir=args.bundle) if args.run.exists() else Engine.create(args.run, config, args.bundle, adapter)
    state = engine.search(startup_seconds=startup)
    return {"run": str(args.run), "phase": state["phase"], "mode": state["mode"], "runs_completed": len(state["runs"])}


def audit(args) -> dict:
    status = verify_run(args.run)
    if status["mode"] == "demo":
        raise ValueError("The demo command audits synthetic fixtures without a real backend")
    if status["phase"] != "frozen":
        raise IntegrityError("Audit needs an already frozen run")
    config = read_json(args.run / "config.json")
    adapter, startup = real_adapter(args, config)
    Engine(args.run, adapter=adapter, bundle_dir=args.bundle).audit(startup_seconds=startup)
    return verify_run(args.run)


def analyze(args) -> dict:
    status = verify_run(args.run)
    frozen_analysis = read_json(args.run / "config.json")["engine"]["analysis"]
    if status["mode"] == "formal" and frozen_analysis != {"bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed}:
        raise IntegrityError("Formal analysis settings must match those frozen before audit")
    report = analyze_run(args.run, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    if args.out.resolve() == args.run.resolve():
        raise ValueError("Analysis output must be a separate directory")
    write_report(report, args.out)
    return {"report": str(args.out / "report.md"), "json": str(args.out / "analysis.json"),
            "mode": status["mode"], "primary": report["primary"]}


def demo(args) -> dict:
    """Exercise the real coordinator on deliberately synthetic score fixtures."""
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    tasks = [{"task_id": f"Synthetic/{i}", "source": "synthetic", "family_id": f"synthetic-family-{i}",
              "prompt": f"SYNTHETIC PIPELINE FIXTURE {i}; not a benchmark problem.", "entry_point": "fixture",
              "canonical_code": "", "inputs": [[i]], "expected": [i], "public_examples": [], "atol": 0.0}
             for i in range(40)]
    bundle, run = args.out / "bundle", args.out / "run"
    config = profile(read_json(args.config), "demo")
    config["engine"]["analysis"] = {"bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed}
    partitions = partition_tasks(tasks, 12, 3, 4, 16, args.seed)
    write_bundle(partitions, bundle, {"purpose": "synthetic-pipeline-demo", "oracle_precomputed": False})
    adapter = SyntheticAdapter()
    engine = Engine.create(run, config, bundle, adapter)
    engine.search()
    engine.freeze()
    engine.audit()
    status = verify_run(run)
    report = analyze_run(run, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.seed)
    write_report(report, args.out / "report")
    return {"mode": "demo", "status": "SYNTHETIC ONLY: no GPU, no generated program execution, no benchmark performance claim",
            "run": str(run), "report": str(args.out / "report" / "report.md"),
            "integrity": status, "study_runs": len(engine.state["runs"]), "audit_tasks": 16}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rsi-audit", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect dependencies without starting an experiment")
    prep = commands.add_parser("prepare", help="Verify pinned data and precompute canonical outputs inside Docker")
    prep.add_argument("--assets", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--config", type=Path, default=Path("configs/study.json"))
    prep.add_argument("--lock", type=Path, default=Path("evidence/baseline-lock.json"))
    prep.add_argument("--family-map", type=Path)
    prep.add_argument("--image", required=True)
    prep.add_argument("--seed", type=int, default=20261028)
    pilot = commands.add_parser("pilot-bundle", help="Construct a 40-task rehearsal exclusively from parent dev data")
    pilot.add_argument("--bundle", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.add_argument("--seed", type=int, default=20261028)
    run = commands.add_parser("search", help="Run or resume a bounded pilot/formal search")
    run.add_argument("--config", type=Path, default=Path("configs/study.json"))
    run.add_argument("--mode", choices=("pilot", "formal"), required=True)
    audit_parser = commands.add_parser("audit", help="Unlock audit data only after all search checkpoints are frozen")
    for target in (run, audit_parser):
        target.add_argument("--run", type=Path, required=True)
        target.add_argument("--bundle", type=Path, required=True)
        target.add_argument("--image", required=True)
        target.add_argument("--model-cache", type=Path, required=True)
        target.add_argument("--device", default="cuda:0")
    for name in ("freeze", "verify"):
        target = commands.add_parser(name)
        target.add_argument("--run", type=Path, required=True)
    analysis = commands.add_parser("analyze", help="Report final checkpoints and paired seed/family uncertainty")
    analysis.add_argument("--run", type=Path, required=True)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--bootstrap-samples", type=int, default=2000)
    analysis.add_argument("--bootstrap-seed", type=int, default=20261028)
    demonstration = commands.add_parser("demo", help="End-to-end synthetic fixture; does not load models or Docker")
    demonstration.add_argument("--out", type=Path, required=True)
    demonstration.add_argument("--config", type=Path, default=Path("configs/study.json"))
    demonstration.add_argument("--seed", type=int, default=20261028)
    demonstration.add_argument("--bootstrap-samples", type=int, default=2000)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "freeze":
            frozen = Engine(args.run).freeze()
            result = {"phase": "frozen", "run": str(args.run), "mode": frozen["mode"],
                      "message": "Archive freeze.json externally before starting audit"}
        elif args.command == "verify":
            result = verify_run(args.run)
        else:
            result = {"prepare": prepare, "pilot-bundle": make_pilot, "search": search,
                      "audit": audit, "analyze": analyze, "demo": demo}[args.command](args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
