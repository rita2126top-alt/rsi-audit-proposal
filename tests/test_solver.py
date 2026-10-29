"""Solver privacy and budget tests; candidate source is NEVER executed here."""

import copy
import importlib
import json
import os
import types
import unittest
from unittest import mock

from rsi_audit.backend import Generation, TransformersBackend
from rsi_audit.common import BudgetExceeded
from rsi_audit.policy import Policy
from rsi_audit.solver import Solver, SyntheticAdapter
from rsi_audit.wire import encode


class FakeBackend:
    """Check allowances, then return prepared text and usage, without inference."""

    def __init__(self, generations):
        self.generations = list(generations)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        if kwargs.get("max_seconds") is not None and kwargs["max_seconds"] <= 0:
            raise BudgetExceeded("Phase time exhausted")
        if kwargs["max_input_tokens"] <= 0 or kwargs["max_new_tokens"] <= 0:
            raise BudgetExceeded("No token allowance remains")
        value = self.generations[0]
        if isinstance(value, Exception):
            self.generations.pop(0)
            raise value
        if value.input_tokens > kwargs["max_input_tokens"]:
            raise BudgetExceeded("Input budget exhausted before inference")
        if value.output_tokens > kwargs["max_new_tokens"]:
            raise AssertionError("Prepared fake response violates the backend allowance")
        self.generations.pop(0)
        return value


class FakeExecutor:
    """Record supplied source/inputs and return inert, predeclared outputs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def execute(self, code, entry_point, inputs):
        self.calls.append({"code": code, "entry_point": entry_point, "inputs": copy.deepcopy(inputs)})
        if not self.responses:
            raise AssertionError("Unexpected executor call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"status": "ok", "outputs": [encode(x) for x in response],
                "wall_seconds": 0.1, "cpu_seconds": 0.02}


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.config = {"solver": {"max_calls_per_task": 2,
                                  "max_total_input_tokens_per_task": 100,
                                  "max_total_output_tokens_per_task": 10}}
        self.task = {"task_id": "private-id-never-in-prompts", "source": "generic", "family_id": "f",
                     "entry_point": "f", "prompt": "Return the input. Public example: f(1) == 1.",
                     "public_examples": [{"args": [1], "kwargs": {}, "expected": 1}],
                     "inputs": [["HIDDEN_INPUT_SENTINEL"]], "expected": [encode("HIDDEN_EXPECTED_SENTINEL")],
                     "atol": 0.0, "canonical_code": "CANONICAL_SECRET_SENTINEL"}
        self.policy = Policy("Implement the function.", "detailed", "syntax_then_examples")

    @staticmethod
    def generation(text="def f(x):\n    return x\n", inputs=20, outputs=3):
        return Generation(text=text, input_tokens=inputs, output_tokens=outputs, seconds=0.2)

    def test_hidden_inputs_answers_and_identity_never_enter_repair(self):
        backend = FakeBackend([self.generation(), self.generation()])
        executor = FakeExecutor([[0], [1], ["HIDDEN_EXPECTED_SENTINEL"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertTrue(result["passed"])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(executor.calls[0]["inputs"], [{"args": [1], "kwargs": {}}])
        self.assertEqual(executor.calls[1]["inputs"], [{"args": [1], "kwargs": {}}])
        self.assertEqual(executor.calls[-1]["inputs"], [["HIDDEN_INPUT_SENTINEL"]])
        prompts = json.dumps([call["messages"] for call in backend.calls])
        for secret in ("HIDDEN_INPUT_SENTINEL", "HIDDEN_EXPECTED_SENTINEL",
                       "CANONICAL_SECRET_SENTINEL", "private-id-never-in-prompts"):
            self.assertNotIn(secret, prompts)
        self.assertIn("expected", backend.calls[1]["messages"][-1]["content"])

    def test_hidden_failure_does_not_trigger_another_generation(self):
        backend = FakeBackend([self.generation()])
        executor = FakeExecutor([[1], ["wrong-hidden-answer"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "wrong_answer")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(len(backend.calls), 1)

    def test_phase_expiry_during_generation_does_not_become_a_wrong_answer(self):
        backend = FakeBackend([self.generation()])
        executor = FakeExecutor([])
        remaining = iter([1.0, -0.01])
        with self.assertRaises(BudgetExceeded) as caught:
            Solver(backend, executor, self.config).solve(self.policy, self.task, remaining_seconds=lambda: next(remaining))
        self.assertEqual(caught.exception.usage["input_tokens"], 20)
        self.assertEqual(caught.exception.usage["output_tokens"], 3)
        self.assertEqual(executor.calls, [])

    def test_total_input_and_output_allowances_are_cumulative(self):
        backend = FakeBackend([self.generation(inputs=40, outputs=3), self.generation(inputs=60, outputs=7)])
        executor = FakeExecutor([[0], [1], ["HIDDEN_EXPECTED_SENTINEL"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertEqual([call["max_input_tokens"] for call in backend.calls], [100, 60])
        self.assertEqual([call["max_new_tokens"] for call in backend.calls], [5, 7])
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 10)
        self.assertEqual(result["model_calls"], 2)

    def test_repair_rejected_before_inference_when_total_input_budget_is_used(self):
        backend = FakeBackend([self.generation(inputs=70), self.generation(inputs=40)])
        executor = FakeExecutor([[0], ["HIDDEN_EXPECTED_SENTINEL"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertEqual(backend.calls[1]["max_input_tokens"], 30)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["input_tokens"], 70)
        self.assertIn("budget", result["stop_reason"].lower())
        self.assertEqual(len(executor.calls), 2)

    def test_repeated_public_failure_still_stops_after_two_calls(self):
        backend = FakeBackend([self.generation(), self.generation()])
        executor = FakeExecutor([[0], [0], ["wrong-hidden-answer"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(len(backend.calls), 2)
        self.assertFalse(result["passed"])

    def test_one_call_mode_receives_entire_output_cap(self):
        self.config["solver"]["max_calls_per_task"] = 1
        backend = FakeBackend([self.generation(outputs=10)])
        executor = FakeExecutor([[0], ["wrong-hidden-answer"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertEqual(backend.calls[0]["max_new_tokens"], 10)
        self.assertEqual(result["model_calls"], 1)
        self.assertFalse(result["passed"])

    def test_syntax_error_is_not_executed_and_can_be_repaired(self):
        backend = FakeBackend([self.generation(text="def f(\n"), self.generation()])
        executor = FakeExecutor([[1], ["HIDDEN_EXPECTED_SENTINEL"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertTrue(result["passed"])
        self.assertIn("syntax", backend.calls[1]["messages"][-1]["content"].lower())
        self.assertTrue(all("def f(\n" != call["code"] for call in executor.calls))

    def test_missing_function_fails_without_executing_candidate(self):
        backend = FakeBackend([self.generation(text="x = 1"), self.generation(text="x = 2")])
        executor = FakeExecutor([])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "invalid_program")
        self.assertEqual(executor.calls, [])

    def test_no_public_examples_does_not_fall_back_to_hidden_feedback(self):
        self.task["public_examples"] = []
        backend = FakeBackend([self.generation()])
        executor = FakeExecutor([["wrong-hidden-answer"]])
        result = Solver(backend, executor, self.config).solve(self.policy, self.task)
        self.assertFalse(result["passed"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(executor.calls), 1)

    def test_infrastructure_failure_propagates_instead_of_becoming_model_score(self):
        backend = FakeBackend([self.generation()])
        executor = FakeExecutor([RuntimeError("Docker daemon unavailable")])
        with self.assertRaisesRegex(RuntimeError, "Docker daemon"):
            Solver(backend, executor, self.config).solve(self.policy, self.task)

    def test_phase_time_exhaustion_propagates_without_starting_execution(self):
        backend = FakeBackend([self.generation()])
        executor = FakeExecutor([])
        with self.assertRaises(BudgetExceeded):
            Solver(backend, executor, self.config).solve(self.policy, self.task, remaining_seconds=lambda: 0)
        self.assertEqual(executor.calls, [])

    def test_repair_order_changes_diagnostic_instruction_order(self):
        repair_messages = []
        for order in ("syntax_then_examples", "examples_then_syntax"):
            backend = FakeBackend([self.generation(text="def f(\n"), self.generation()])
            executor = FakeExecutor([[1], ["HIDDEN_EXPECTED_SENTINEL"]])
            policy = Policy("Implement the function.", "compact", order)
            Solver(backend, executor, self.config).solve(policy, self.task)
            repair_messages.append(backend.calls[1]["messages"][-1]["content"])
        self.assertNotEqual(repair_messages[0], repair_messages[1])
        for message in repair_messages:
            self.assertIn("syntax", message.lower())
            self.assertIn("public", message.lower())

    def test_synthetic_adapter_is_explicit_and_uses_no_executor_or_model(self):
        adapter = SyntheticAdapter()
        result = adapter.solve(self.policy, self.task)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(adapter.gpu_count, 0)
        self.assertIn("SYNTHETIC", result["code"])


class OfflineBackendTests(unittest.TestCase):
    revision = "c" * 40

    def modules(self, *, torch_version="2.6.0+cu124", transformers_version="4.51.3", cuda_available=True):
        tokenizer = types.SimpleNamespace(chat_template="fixture", encode=lambda text, **kwargs: text.split())
        model = mock.Mock()
        model.to.return_value = model
        model.eval.return_value = model
        torch = types.ModuleType("torch")
        torch.__version__ = torch_version
        torch.bfloat16 = "test-bfloat16"
        torch.version = types.SimpleNamespace(cuda="test-cuda")
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available, get_device_name=lambda device: "TEST GPU")
        transformers = types.ModuleType("transformers")
        transformers.__version__ = transformers_version
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=tokenizer))
        transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=model))
        return torch, transformers

    def test_pinned_model_and_tokenizer_load_only_local_safe_assets(self):
        torch, transformers = self.modules()
        with mock.patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
                mock.patch("rsi_audit.backend.version", return_value="test-version"), \
                mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}):
            backend = TransformersBackend("Qwen/test", self.revision, cache_dir="test-cache")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            for loader in (transformers.AutoTokenizer.from_pretrained, transformers.AutoModelForCausalLM.from_pretrained):
                kwargs = loader.call_args.kwargs
                self.assertTrue(kwargs["local_files_only"])
                self.assertFalse(kwargs["trust_remote_code"])
                self.assertEqual(kwargs["revision"], self.revision)
            self.assertTrue(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["use_safetensors"])
            self.assertEqual(backend.gpu_count, 1)

    def test_unpinned_revision_is_rejected_before_model_loading(self):
        for revision in ("main", "v1", "z" * 40, "1" * 39):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                TransformersBackend("Qwen/test", revision)

    def test_wrong_dependency_version_fails_closed_before_loading(self):
        for kwargs in ({"torch_version": "2.5.0"}, {"transformers_version": "4.52.0"}):
            torch, transformers = self.modules(**kwargs)
            with self.subTest(kwargs=kwargs), \
                    mock.patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
                    mock.patch.dict(os.environ), self.assertRaisesRegex(RuntimeError, "requires transformers"):
                TransformersBackend("Qwen/test", self.revision)
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_missing_gpu_does_not_silently_use_cpu(self):
        torch, transformers = self.modules(cuda_available=False)
        with mock.patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
                mock.patch.dict(os.environ), self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
            TransformersBackend("Qwen/test", self.revision)
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_missing_optional_dependencies_have_actionable_error(self):
        with mock.patch.dict("sys.modules", {"torch": None, "transformers": None}), \
                mock.patch.dict(os.environ), self.assertRaisesRegex(RuntimeError, "Install the gpu extra"):
            TransformersBackend("Qwen/test", self.revision)

    def test_importing_backend_does_not_import_torch_or_transformers(self):
        import rsi_audit.backend as module
        original_import = __import__
        attempted = []
        def checked_import(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                attempted.append(name)
                raise AssertionError("Optional GPU package imported at module import time")
            return original_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=checked_import):
            importlib.reload(module)
        self.assertEqual(attempted, [])


if __name__ == "__main__":
    unittest.main()
