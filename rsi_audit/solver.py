"""Bounded coding trajectories. Hidden test outcomes never enter repair prompts."""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
from importlib.metadata import version

from .common import BudgetExceeded, ensure_identifier
from .policy import Policy


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[0] if blocks else text).strip() + "\n"


def syntax_feedback(code: str, entry_point: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        return f"Syntax error: {str(exc)[:400]}"
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point for node in tree.body):
        return f"Define the complete top-level function {entry_point}."
    return None


class Solver:
    def __init__(self, backend, executor, config: dict):
        self.backend, self.executor, self.config = backend, executor, config

    def solve(self, policy: Policy, task: dict, *, remaining_seconds=None) -> dict:
        from .scoring import score_outputs
        from .wire import encode

        entry_point = ensure_identifier(task["entry_point"])
        cap = self.config["solver"]
        max_calls = cap["max_calls_per_task"]
        if max_calls not in (1, 2):
            raise ValueError("This implementation supports one or two solver calls")
        usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0,
                 "model_seconds": 0.0, "execution_seconds": 0.0, "cpu_seconds": None}
        messages = [{"role": "system", "content": (
            policy.strategy_prompt + "\nFrozen execution contract: Python standard library only. "
            "Return a complete function and required imports. Do not access files, the network or external services. "
            "Return one Python code block; no test-score claims."
        )}, {"role": "user", "content": task["prompt"]}]
        code, feedback, stop_reason = "", "", None
        for attempt in range(max_calls):
            input_left = cap["max_total_input_tokens_per_task"] - usage["input_tokens"]
            output_left = cap["max_total_output_tokens_per_task"] - usage["output_tokens"]
            allowance = min(output_left, (cap["max_total_output_tokens_per_task"] + max_calls - 1) // max_calls) if attempt == 0 else output_left
            seconds_left = remaining_seconds() if remaining_seconds else None
            if seconds_left is not None and seconds_left <= 0:
                raise BudgetExceeded("Phase budget exhausted before model generation")
            try:
                generation = self.backend.generate(
                    messages, max_input_tokens=input_left, max_new_tokens=allowance,
                    temperature=0.0, top_p=1.0, seed=0,
                    max_seconds=seconds_left,
                )
            except BudgetExceeded as exc:
                stop_reason = str(exc)
                break
            usage["input_tokens"] += generation.input_tokens
            usage["output_tokens"] += generation.output_tokens
            usage["model_calls"] += 1
            usage["model_seconds"] += generation.seconds
            if remaining_seconds is not None and remaining_seconds() <= 0:
                error = BudgetExceeded("Phase time expired during model generation; incomplete task is not scored")
                error.usage = dict(usage)
                raise error
            code = extract_code(generation.text)
            syntax = syntax_feedback(code, entry_point)
            example_feedback = None
            examples = task.get("public_examples", [])
            if not syntax and examples:
                if remaining_seconds is not None and remaining_seconds() <= 0:
                    raise BudgetExceeded("Phase budget exhausted before public example execution")
                execution = self.executor.execute(code, entry_point, [
                    {"args": example["args"], "kwargs": example.get("kwargs", {})} for example in examples
                ])
                usage["execution_seconds"] += execution["wall_seconds"]
                public_task = {"task_id": "public-examples", "source": "generic", "entry_point": entry_point,
                               "inputs": [x["args"] for x in examples], "expected": [encode(x["expected"]) for x in examples], "atol": 0.0}
                public_result = score_outputs(public_task, execution)
                if public_result["status"].startswith("unsupported"):
                    raise RuntimeError("Public example scoring backend is unavailable")
                if not public_result["passed"]:
                    example_feedback = "The solution did not satisfy the examples explicitly shown in the task prompt."
                    if policy.feedback_style == "detailed":
                        example_feedback += " Recheck these public examples: " + repr(examples)[:3000]
            if syntax or example_feedback:
                syntax_part = "Syntax/contract check: " + (syntax or "passed; preserve the complete entry-point function.")
                example_part = "Public example check: " + (example_feedback or (
                    "not run until syntax is valid." if syntax else "no structured examples available."))
                parts = [syntax_part, example_part] if policy.repair_order == "syntax_then_examples" else [example_part, syntax_part]
                feedback = "\n".join(parts)
            else:
                feedback = ""
            if not feedback or attempt + 1 == max_calls:
                break
            messages += [{"role": "assistant", "content": generation.text},
                         {"role": "user", "content": feedback + "\nReturn a corrected complete solution. Hidden tests and their results are unavailable."}]

        if not code:
            result = {"passed": False, "status": "input_budget_exceeded", "tests_run": 0}
        elif syntax_feedback(code, entry_point):
            result = {"passed": False, "status": "invalid_program", "tests_run": 0}
        else:
            if remaining_seconds is not None and remaining_seconds() <= 0:
                raise BudgetExceeded("Phase budget exhausted before independent scoring")
            execution = self.executor.execute(code, entry_point, task["inputs"])
            usage["execution_seconds"] += execution["wall_seconds"]
            result = score_outputs(task, execution)
            if result["status"].startswith("unsupported"):
                raise RuntimeError(f"Unsupported scoring for {task['task_id']}: {result['status']}")
        return dict(result, **usage, code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    public_feedback=feedback, stop_reason=stop_reason, synthetic=False)


class RealAdapter:
    synthetic = False

    def __init__(self, backend, executor, config: dict):
        self.backend, self.executor, self.config = backend, executor, config
        self.solver = Solver(backend, executor, config)
        self.gpu_count = backend.gpu_count
        self.identity = {"model": backend.identity, "executor": executor.identity, "synthetic": False,
                         "python": platform.python_version(), "numpy": version("numpy"),
                         "execution_limits": config["engine"]["limits"]}

    def count_tokens(self, text: str) -> int:
        return self.backend.count_tokens(text)

    def propose(self, parent: Policy, failures: list[dict], *, seed: int, max_seconds=None) -> dict:
        spec = self.config["search"]
        messages = [{"role": "system", "content": (
            "Improve a persistent Python coding-agent strategy using development failures only. "
            "Return exactly one JSON object with strategy_prompt, feedback_style, repair_order. "
            "feedback_style must be compact or detailed; repair_order must be syntax_then_examples or examples_then_syntax. "
            "Do not add fields, tools, code plugins, budgets, data access or scoring changes. "
            f"strategy_prompt must fit {spec['strategy_prompt_max_tokens']} tokens."
        )}, {"role": "user", "content": json.dumps({"current_policy": parent.to_dict(), "development_failures": failures[:6]}, ensure_ascii=False)}]
        generation = self.backend.generate(
            messages, max_input_tokens=8192, max_new_tokens=spec["proposer_max_output_tokens"],
            temperature=spec["proposer_temperature"], top_p=spec["proposer_top_p"], seed=seed, max_seconds=max_seconds,
        )
        return {"text": generation.text, "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens, "model_calls": 1, "model_seconds": generation.seconds}

    def solve(self, policy: Policy, task: dict, *, remaining_seconds=None) -> dict:
        return self.solver.solve(policy, task, remaining_seconds=remaining_seconds)


class SyntheticAdapter:
    """Deterministic synthetic outcomes ONLY for offline pipeline verification."""
    synthetic = True
    gpu_count = 0
    identity = {"model": "synthetic-fixture-v1", "executor": "none", "synthetic": True}

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    def propose(self, parent: Policy, failures: list[dict], *, seed: int, max_seconds=None) -> dict:
        candidate = Policy(f"SYNTHETIC FIXTURE ONLY: strategy {seed}; parent {parent.sha256[:12]}",
                           "detailed" if seed % 2 else "compact", "syntax_then_examples")
        text = json.dumps(candidate.to_dict())
        return {"text": text, "input_tokens": 0, "output_tokens": 0, "model_calls": 0, "model_seconds": 0.0}

    def solve(self, policy: Policy, task: dict, *, remaining_seconds=None) -> dict:
        value = int(hashlib.sha256((policy.sha256 + ":" + task["task_id"]).encode()).hexdigest()[:8], 16)
        passed = value % 100 < 60
        code = "# SYNTHETIC FIXTURE: no model or candidate program was executed.\n"
        return {"passed": passed, "status": "synthetic_pass" if passed else "synthetic_fail", "tests_run": 0,
                "input_tokens": 0, "output_tokens": 0, "model_calls": 0, "model_seconds": 0.0,
                "execution_seconds": 0.0, "cpu_seconds": None, "code": code,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "public_feedback": "",
                "stop_reason": None, "synthetic": True}
