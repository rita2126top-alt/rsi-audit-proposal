#!/usr/bin/env python3
"""Validate proposal artifacts without fetching data or running experiments."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    form = read_json("submission/form.public.json")
    study = read_json("configs/study.json")
    evidence = read_json("evidence/baseline-lock.json")
    sources = read_json("evidence/sources.json")
    required = {
        "direction", "question", "baseline", "evaluation", "plan",
        "gpu_type", "gpu_trial", "gpu_full", "portfolio", "contact",
    }
    require(set(form) == required, "Unexpected form fields")
    bounds = {
        "question": (10, 300), "baseline": (1, 600),
        "evaluation": (10, 600), "plan": (20, 1000),
        "gpu_type": (1, 40), "portfolio": (1, 300), "contact": (0, 100),
    }
    for key, (lower, upper) in bounds.items():
        require(isinstance(form[key], str), f"{key} must be text")
        count = len(form[key].encode("utf-16-le")) // 2
        require(lower <= count <= upper, f"{key}: {count} outside {lower}..{upper}")
        print(f"{key}: {count}/{upper} UTF-16 units")
    require(form["direction"] == "eval", "Wrong program direction")
    require(form["contact"] == "", "Private contact must not be committed")
    require("https://" in form["baseline"], "Baseline needs a source link")
    require(form["portfolio"].startswith("https://github.com/"), "Missing public portfolio")
    budget = study["budget"]
    require(form["gpu_trial"] == budget["trial_gpu_hours"] == 48, "Trial budget mismatch")
    require(form["gpu_full"] == budget["full_gpu_hours_including_trial"] == 480, "Full budget mismatch")
    require(sum(budget["phase_gpu_hours"].values()) == form["gpu_full"], "Phase hours do not sum")
    require(budget["trial_days"] == 7 and budget["full_days"] == 28, "Program timing mismatch")
    require(study["status"] == "proposal_not_executed", "This package contains no executed experiment")
    require(study["model"]["revision"] == evidence["model"]["revision"], "Model pin mismatch")
    require(study["evaluator"]["commit"] == evidence["proposed_evaluator"]["commit"], "Evaluator pin mismatch")
    require(sum(x["task_count_verified"] for x in evidence["datasets"]) == study["data"]["raw_tasks"], "Raw dataset size mismatch")
    data, search, gate, audit = (study[k] for k in ("data", "search", "gate", "audit"))
    require(data["target_dev"] + data["target_confirm"] + data["target_audit"] == data["raw_tasks"], "Target split does not sum")
    require(data["actual_split_manifest"] is None, "Do not imply an actual split was made")
    require(gate["panel_tasks"] * search["rounds"] == gate["paired_comparisons_per_arm_seed"] == gate["terminal_tasks"], "Gate allowances differ")
    require(gate["reuse_fresh_net_successes_required"] / gate["panel_tasks"] == gate["terminal_net_successes_required"] / gate["terminal_tasks"], "Gate margins differ")
    per_run = data["target_dev"] * (1 + search["rounds"] * search["candidates_per_round"])
    per_run += 2 * gate["paired_comparisons_per_arm_seed"]
    per_run += audit["max_checkpoints_per_arm_seed"] * data["target_audit"]
    runs = len(search["arms"]) * len(search["proposer_seeds"])
    require(per_run * runs == budget["uncached_task_rollouts_upper_bound"] == 64836, "Rollout budget mismatch")
    require(runs * search["rounds"] * search["candidates_per_round"] == budget["proposer_calls_upper_bound"], "Proposer budget mismatch")
    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    require(len(keys) == len(set(keys)) == 9, "Expected 9 unique bibliography entries")
    require(set(keys) == {x["key"] for x in sources["sources"]}, "Bibliography and sources mismatch")
    for md in ROOT.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            require((md.parent / relative).is_file(), f"Broken local link: {md.name}: {relative}")
        for group in re.findall(r"\[((?:[a-z]+20\d{2}[a-z0-9]+)(?:; [a-z]+20\d{2}[a-z0-9]+)*)\]", text):
            for key in group.split("; "):
                require(key in keys, f"Unknown citation key: {key}")
    require((ROOT / "CITATION.cff").is_file(), "Missing CFF")
    require((ROOT / "LICENSE").is_file(), "Missing license")
    print(f"OK: {len(keys)} references; {per_run * runs:,} planned uncached trajectories; 48/480 GPU hours.")
    print("Public proposal package validated. Private contact is intentionally empty; no website submission or GPU experiment was performed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
