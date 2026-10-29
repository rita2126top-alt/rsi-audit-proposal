"""CPU boundary tests plus explicitly opted-in Docker integration tests.

Run integration tests only after building the image, for example:
RSI_AUDIT_TEST_IMAGE=rsi-audit-worker:local python -m unittest discover -s tests
No candidate source is executed on the host by these tests.
"""

import json
import math
import os
import shutil
import subprocess
import unittest
from unittest import mock

from rsi_audit import wire
from rsi_audit.execution import DockerExecutor, _Deadline, _OutputLimit
from rsi_audit.scoring import score_outputs

IMAGE_ID = "sha256:" + "a" * 64


def docker_reply(command, **_):
    if command[1] == "version":
        stdout = b"27.0.0"
    elif command[1:3] == ["image", "inspect"]:
        stdout = json.dumps({"Id": IMAGE_ID, "Os": "linux", "Config": {
            "Labels": {"org.rsi-audit.worker.protocol": "1"}}}).encode()
    elif command[1] == "inspect":
        stdout = b'{"OOMKilled":true}'
    else:
        stdout = b""
    return subprocess.CompletedProcess(command, 0, stdout, b"")


class WireTests(unittest.TestCase):
    def test_nested_round_trip_and_determinism(self):
        value = {"tuple": (1, 2.5), "set": {"z", "a"}, "complex": 1 + 2j,
                 "dict": {(1, 2): {"x": None}}, "frozen": frozenset({3, 2})}
        encoded = wire.encode(value)
        self.assertEqual(wire.decode(encoded), value)
        reordered = dict(reversed(list(value.items())))
        self.assertEqual(encoded, wire.encode(reordered))
        self.assertEqual(wire.encode({1, 2, 3}), wire.encode({3, 1, 2}))

    def test_scalar_types_survive(self):
        for value in (None, True, False, 0, -13, 2 ** 120, -0.0, "中文", [1], (1,)):
            with self.subTest(value=value):
                decoded = wire.decode(wire.encode(value))
                self.assertIs(type(decoded), type(value))
                self.assertEqual(decoded, value)

    def test_nonfinite_requires_explicit_tag(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            encoded = wire.encode(value)
            self.assertEqual(encoded["t"], "nonfinite")
            decoded = wire.decode(encoded)
            self.assertTrue(math.isnan(decoded) if math.isnan(value) else decoded == value)
            json.dumps(encoded, allow_nan=False)
        with self.assertRaises(wire.WireError):
            wire.decode({"t": "float", "v": float("nan")})
        for text in ('{"a":NaN}', '{"a":Infinity}', '{"a":1,"a":2}'):
            with self.assertRaises(wire.WireError):
                wire.strict_json_loads(text)

    def test_rejects_arbitrary_objects_cycles_and_shapes(self):
        class Trap:
            def __repr__(self):
                raise AssertionError("must not call a candidate representation")
        cyclic = []
        cyclic.append(cyclic)
        for value in (Trap(), cyclic):
            with self.assertRaises(wire.WireError):
                wire.encode(value)
        invalid = [
            {"t": "constructor", "v": "os.system"},
            {"t": "none", "v": 1}, {"t": "bool", "v": 1},
            {"t": "int", "v": "01"}, {"t": "int", "v": "-0"},
            {"t": "complex", "v": [wire.encode("a"), wire.encode(1)]},
            {"t": "set", "v": [wire.encode([])]},
            {"t": "set", "v": [wire.encode(1), wire.encode(True)]},
            {"t": "dict", "v": [[wire.encode("a"), wire.encode(1)],
                                   [wire.encode("a"), wire.encode(2)]]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(wire.WireError):
                wire.decode(value)

    def test_size_depth_node_limits(self):
        with self.assertRaises(wire.WireError):
            wire.encode([1, 2], max_nodes=2)
        with self.assertRaises(wire.WireError):
            wire.encode([[1]], max_depth=1)
        with self.assertRaises(wire.WireError):
            wire.encode("abc", max_bytes=10)
        with self.assertRaises(wire.WireError):
            wire.decode(wire.encode([[1]]), max_depth=1)
        with self.assertRaises(wire.WireError):
            wire.decode(wire.encode("abc"), max_bytes=10)


class ExecutorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.which = mock.patch("rsi_audit.execution.shutil.which", return_value="/usr/bin/docker")
        self.run = mock.patch("rsi_audit.execution.subprocess.run", side_effect=docker_reply)
        self.which.start()
        self.runner = self.run.start()
        self.addCleanup(self.which.stop)
        self.addCleanup(self.run.stop)
        self.executor = DockerExecutor("worker:mutable-tag")

    def test_immutable_identity_and_hardened_command(self):
        self.assertEqual(self.executor.identity, IMAGE_ID)
        command = self.executor._build_command("rsi-audit-test")
        for flag, value in (("--network", "none"), ("--cap-drop", "ALL"),
                            ("--security-opt", "no-new-privileges=true"),
                            ("--user", "65534:65534"), ("--pull", "never"),
                            ("--pids-limit", "32"), ("--memory", "512m"),
                            ("--memory-swap", "512m"), ("--cpus", "1.0")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--read-only", command)
        self.assertIn(IMAGE_ID, command)
        self.assertNotIn("worker:mutable-tag", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--volume", command)
        self.assertNotIn("--mount", command)
        self.assertNotIn("--env-file", command)

    def test_stdin_payload_only_contains_code_entry_and_inputs(self):
        response = json.dumps({"status": "ok", "outputs": [wire.encode(3)], "cpu_seconds": 0.01}).encode()
        with mock.patch.object(self.executor, "_run_container", return_value=(0, response, b"")) as invoke:
            result = self.executor.execute("def f(x): return x", "f", [[{1, 2}]])
        payload = json.loads(invoke.call_args.args[1])
        self.assertEqual(set(payload), {"code", "entry_point", "inputs"})
        self.assertEqual(wire.decode(payload["inputs"]), [[{1, 2}]])
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["wall_seconds"], 0)
        self.assertTrue(any(call.args[0][1:3] == ["rm", "--force"]
                            for call in self.runner.call_args_list))

    def test_timeout_and_output_limit_always_cleanup(self):
        for error, status in ((_Deadline(), "timeout"), (_OutputLimit(), "protocol_error")):
            with self.subTest(status=status):
                with mock.patch.object(self.executor, "_run_container", side_effect=error), \
                        mock.patch.object(self.executor, "_cleanup") as cleanup:
                    result = self.executor.execute("def f(): return 1", "f", [[]])
                self.assertEqual(result["status"], status)
                self.assertEqual(result["outputs"], [])
                cleanup.assert_called_once()

    def test_infrastructure_failure_raises_and_cleans(self):
        with mock.patch.object(self.executor, "_run_container", return_value=(125, b"", b"daemon down")), \
                mock.patch.object(self.executor, "_cleanup") as cleanup:
            with self.assertRaises(RuntimeError):
                self.executor.execute("def f(): return 1", "f", [[]])
        cleanup.assert_called_once()
        with mock.patch.object(self.executor, "_run_container", side_effect=RuntimeError("spawn failed")), \
                mock.patch.object(self.executor, "_cleanup") as cleanup:
            with self.assertRaises(RuntimeError):
                self.executor.execute("def f(): return 1", "f", [[]])
        cleanup.assert_called_once()

    def test_confirmed_oom_is_candidate_failure(self):
        with mock.patch.object(self.executor, "_run_container", return_value=(137, b"", b"")):
            self.assertEqual(self.executor.execute("def f(): return 1", "f", [[]])["status"],
                             "candidate_error")

    def test_cleanup_failure_aborts_instead_of_returning_score(self):
        response = b'{"status":"ok","outputs":[{"t":"int","v":"1"}],"cpu_seconds":0}'
        with mock.patch.object(self.executor, "_run_container", return_value=(0, response, b"")), \
                mock.patch.object(self.executor, "_cleanup", side_effect=RuntimeError("daemon down")):
            with self.assertRaises(RuntimeError):
                self.executor.execute("def f(): return 1", "f", [[]])

    def test_protocol_does_not_accept_self_reported_pass(self):
        for response in (
            b'{"passed":true}',
            b'{"status":"ok","outputs":[],"cpu_seconds":0}',
            b'{"status":"ok","outputs":[{"t":"pickle","v":"payload"}],"cpu_seconds":0}',
            b'{"status":"ok","outputs":[{"t":"int","v":"1"}],"cpu_seconds":NaN}',
            b'{"status":"ok","outputs":[{"t":"int","v":"1"}],"cpu_seconds":0,"passed":true}',
        ):
            self.assertEqual(self.executor._parse_response(response, 1)["status"], "protocol_error")

    def test_invalid_local_inputs_never_launch(self):
        with mock.patch.object(self.executor, "_run_container") as launch:
            for inputs in ([], [1], [{"expected": 1}], [{"args": (), "kwargs": {}}],
                           [{"args": [], "kwargs": {1: 2}}]):
                with self.assertRaises(ValueError):
                    self.executor.execute("def f(): return 1", "f", inputs)
            for entry in ("f()", "__import__", "class", "a.b"):
                with self.assertRaises(ValueError):
                    self.executor.execute("def f(): return 1", entry, [[]])
            launch.assert_not_called()

    def test_missing_docker_and_unlabelled_image_fail_closed(self):
        with mock.patch("rsi_audit.execution.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                DockerExecutor("worker:test")
        with mock.patch.object(DockerExecutor, "_control", side_effect=["27", json.dumps({"Id": IMAGE_ID})]):
            with self.assertRaises(RuntimeError):
                DockerExecutor("worker:test")
        for kwargs in ({"cpus": float("nan")}, {"timeout_seconds": -1}, {"memory_mb": True}):
            with self.assertRaises(ValueError):
                DockerExecutor("worker:test", **kwargs)


@unittest.skipUnless(shutil.which("docker") and os.environ.get("RSI_AUDIT_TEST_IMAGE"),
                     "requires Docker and explicitly configured RSI_AUDIT_TEST_IMAGE")
class DockerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = DockerExecutor(os.environ["RSI_AUDIT_TEST_IMAGE"], timeout_seconds=5)

    def test_real_execution_and_stdout_forgery(self):
        code = 'import os\ndef f(x):\n print("{\\\"passed\\\":true}")\n os.write(1,b"PASS")\n return x+1\n'
        result = self.executor.execute(code, "f", [[3], [5]])
        self.assertEqual(result["status"], "ok")
        self.assertEqual([wire.decode(x) for x in result["outputs"]], [4, 6])
        task = {"task_id": "test", "source": "generic", "entry_point": "f", "atol": 0,
                "inputs": [[3], [5]], "expected": [wire.encode(99), wire.encode(99)]}
        self.assertFalse(score_outputs(task, result)["passed"])

    def test_types_and_invalid_arity(self):
        value = ({1, 3}, (2, 4), 1 + 2j)
        result = self.executor.execute("def f(x): return x", "f", [[value]])
        self.assertEqual(wire.decode(result["outputs"][0]), value)
        result = self.executor.execute("def f(x): return x", "f", [[]])
        self.assertEqual(result["status"], "candidate_error")

    def test_no_network_read_only_and_unprivileged(self):
        code = '''import os, socket
def f():
    blocked = False
    try:
        open('/opt/rsi_audit/probe', 'w').write('x')
    except OSError:
        blocked = True
    s = socket.socket()
    s.settimeout(0.2)
    network = False
    try:
        s.connect(('1.1.1.1', 53))
        network = True
    except OSError:
        pass
    finally:
        s.close()
    return [os.getuid(), blocked, network]
'''
        result = self.executor.execute(code, "f", [[]])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(wire.decode(result["outputs"][0]), [65534, True, False])

    def test_infinite_loop_is_stopped_and_container_removed(self):
        with mock.patch.object(self.executor, "_build_command", wraps=self.executor._build_command) as build:
            result = self.executor.execute("def f():\n while True: pass", "f", [[]])
        self.assertEqual(result["status"], "timeout")
        name = build.call_args.args[0]
        remaining = self.executor._control(["ps", "--all", "--filter", "name=^" + name + "$",
                                            "--format", "{{.Names}}"])
        self.assertEqual(remaining, "")


if __name__ == "__main__":
    unittest.main()
