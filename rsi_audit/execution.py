"""Docker-only execution of generated Python, with host-owned resource limits.

There is deliberately no local Python fallback. Docker daemon/image failures
raise RuntimeError and must abort a run, rather than lowering a model's score.
An image tag is resolved once to an immutable image ID before any candidate is
executed. Docker provides isolation from the host; it is not a VM security
boundary against a kernel exploit. Use a dedicated research host.
"""

from __future__ import annotations

import json
import keyword
import math
import os
import re
import selectors
import shutil
import subprocess
import time
import uuid

from . import wire

MAX_CODE_BYTES = 256 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_REPLY_BYTES = 16 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_INPUTS = 4096
PROTOCOL_LABEL = "org.rsi-audit.worker.protocol"
PROTOCOL_VERSION = "1"


class _Deadline(Exception):
    pass


class _OutputLimit(Exception):
    pass


def _valid_number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if not low <= value <= high:
        raise ValueError(name + " is outside the supported range")


def _validate_calls(inputs):
    if type(inputs) is not list or not 1 <= len(inputs) <= MAX_INPUTS:
        raise ValueError("inputs must contain 1..4096 argument records")
    for record in inputs:
        if type(record) is list:
            args, kwargs = record, {}
        elif type(record) is dict and set(record) <= {"args", "kwargs"} and "args" in record:
            args, kwargs = record["args"], record.get("kwargs", {})
        else:
            raise ValueError("each input must be an args list or {args, kwargs}")
        if type(args) is not list or type(kwargs) is not dict:
            raise ValueError("args must be a list and kwargs a dictionary")
        if len(args) + len(kwargs) > 128:
            raise ValueError("argument count exceeds 128")
        if any(type(key) is not str for key in kwargs):
            raise ValueError("keyword argument names must be strings")


class DockerExecutor:
    """Execute candidates in a local, prebuilt RSI-Audit worker image.

    ``identity`` is the immutable ``sha256:...`` Docker image ID. The supplied
    image must already exist locally and carry the worker protocol label.
    Building/pulling images is an explicit setup step, never part of scoring.
    """

    def __init__(self, image: str, timeout_seconds: float = 30,
                 memory_mb: int = 512, cpus: float = 1.0):
        if type(image) is not str or not image or image.startswith("-") or any(c.isspace() for c in image):
            raise ValueError("image must be a nonempty Docker image reference")
        _valid_number(timeout_seconds, "timeout_seconds", 0.05, 3600)
        _valid_number(cpus, "cpus", 0.05, 64)
        if type(memory_mb) is not int or not 64 <= memory_mb <= 131072:
            raise ValueError("memory_mb must be an integer in [64, 131072]")
        self.timeout_seconds = float(timeout_seconds)
        self.memory_mb = memory_mb
        self.cpus = float(cpus)
        self.docker = shutil.which("docker")
        if not self.docker:
            raise RuntimeError("Docker CLI is unavailable; candidate execution has no host fallback")
        # A server command distinguishes an installed CLI from a working daemon.
        self._control(["version", "--format", "{{.Server.Version}}"])
        raw = self._control(["image", "inspect", "--format", "{{json .}}", image])
        try:
            metadata = wire.strict_json_loads(raw)
            identity = metadata["Id"]
            labels = metadata.get("Config", {}).get("Labels") or {}
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
                raise ValueError("invalid image ID")
            if metadata.get("Os", "linux") != "linux":
                raise ValueError("only Linux worker images are supported")
            if labels.get(PROTOCOL_LABEL) != PROTOCOL_VERSION:
                raise ValueError("image does not have the RSI-Audit worker protocol label")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError("cannot verify the worker image; build docker/Dockerfile first") from exc
        self.identity = identity

    def _control(self, arguments, *, allow_missing_container=False):
        try:
            completed = subprocess.run([self.docker, *arguments], capture_output=True,
                                       timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Docker infrastructure command failed: " + arguments[0]) from exc
        if len(completed.stdout) > 256 * 1024 or len(completed.stderr) > 256 * 1024:
            raise RuntimeError("Docker infrastructure response exceeded its limit")
        if completed.returncode:
            message = completed.stderr.decode("utf-8", errors="replace")
            if allow_missing_container and ("No such container" in message or "No such object" in message):
                return ""
            # Do not copy Docker configuration or daemon paths into research logs.
            raise RuntimeError("Docker infrastructure command returned an error: " + arguments[0])
        try:
            return completed.stdout.decode("utf-8").strip()
        except UnicodeError as exc:
            raise RuntimeError("Docker infrastructure returned non-UTF-8 metadata") from exc

    def _build_command(self, name):
        return [
            self.docker, "run", "--name", name, "--pull", "never", "--interactive",
            "--network", "none", "--ipc", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
            "--user", "65534:65534", "--pids-limit", "32",
            "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
            "--cpus", str(self.cpus), "--ulimit", "nofile=64:64",
            "--ulimit", "core=0:0", "--ulimit", "fsize=1048576:1048576",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
            "--log-driver", "none", "--workdir", "/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONHASHSEED=0",
            "--env", "HOME=/tmp", "--env", "OMP_NUM_THREADS=1",
            "--label", "org.rsi-audit.run=true", "--entrypoint", "python3",
            self.identity, "-I", "/opt/rsi_audit/worker.py",
        ]

    def _cleanup(self, name):
        self._control(["rm", "--force", name], allow_missing_container=True)

    def _run_container(self, command, payload):
        """Bound stdout/stderr in memory while also enforcing the wall deadline."""
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
        except OSError as exc:
            raise RuntimeError("could not start Docker CLI") from exc
        selector = selectors.DefaultSelector()
        stdout, stderr = bytearray(), bytearray()
        offset = 0
        deadline = time.monotonic() + self.timeout_seconds
        try:
            for stream, mode, label in ((process.stdin, selectors.EVENT_WRITE, "stdin"),
                                        (process.stdout, selectors.EVENT_READ, "stdout"),
                                        (process.stderr, selectors.EVENT_READ, "stderr")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, mode, label)
            while selector.get_map():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise _Deadline()
                for key, _ in selector.select(min(left, 0.1)):
                    stream, label = key.fileobj, key.data
                    if label == "stdin":
                        try:
                            n = os.write(stream.fileno(), payload[offset:offset + 65536])
                            offset += n
                        except BrokenPipeError:
                            offset = len(payload)
                        except BlockingIOError:
                            continue
                        if offset == len(payload):
                            selector.unregister(stream)
                            stream.close()
                    else:
                        try:
                            chunk = os.read(stream.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                            continue
                        target, limit = ((stdout, MAX_REPLY_BYTES) if label == "stdout"
                                         else (stderr, MAX_STDERR_BYTES))
                        if len(target) + len(chunk) > limit:
                            raise _OutputLimit()
                        target.extend(chunk)
            try:
                returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise _Deadline() from exc
            return returncode, bytes(stdout), bytes(stderr)
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Docker client could not be stopped") from exc
            for stream in (process.stdin, process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()

    @staticmethod
    def _parse_response(raw, expected_count):
        if type(raw) is not bytes or len(raw) > MAX_REPLY_BYTES:
            return {"status": "protocol_error", "outputs": [], "cpu_seconds": None}
        try:
            message = wire.strict_json_loads(raw.decode("utf-8"))
            if type(message) is not dict or set(message) != {"status", "outputs", "cpu_seconds"}:
                raise wire.WireError("invalid worker response fields")
            if message["status"] not in ("ok", "candidate_error", "protocol_error"):
                raise wire.WireError("invalid worker status")
            outputs = message["outputs"]
            if type(outputs) is not list or len(outputs) > expected_count:
                raise wire.WireError("invalid output count")
            if message["status"] == "ok" and len(outputs) != expected_count:
                raise wire.WireError("missing outputs")
            for output in outputs:
                wire.decode(output, max_nodes=1_000_000, max_bytes=MAX_REPLY_BYTES)
            cpu = message["cpu_seconds"]
            if cpu is not None and (type(cpu) not in (int, float) or
                                    not math.isfinite(cpu) or cpu < 0):
                raise wire.WireError("invalid CPU measurement")
            return message
        except (wire.WireError, ValueError, TypeError, UnicodeError, KeyError):
            return {"status": "protocol_error", "outputs": [], "cpu_seconds": None}

    def execute(self, code: str, entry_point: str, inputs: list) -> dict:
        """Return outputs only. The worker never receives expected answers.

        Invalid local input raises ValueError; unavailable infrastructure raises
        RuntimeError. A timed-out or malformed candidate result cannot pass.
        The deadline includes Docker startup. CPU time is measured by the
        container supervisor; it is unavailable on externally stopped runs.
        """
        if type(code) is not str or not code.strip() or len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError("code must be nonempty UTF-8 text within 256 KiB")
        if (type(entry_point) is not str or not entry_point.isidentifier() or
                keyword.iskeyword(entry_point) or entry_point.startswith("__") or len(entry_point) > 128):
            raise ValueError("entry_point must be an ordinary Python identifier")
        _validate_calls(inputs)
        encoded_inputs = wire.encode(inputs, max_nodes=1_000_000, max_bytes=MAX_PAYLOAD_BYTES)
        payload = json.dumps({"code": code, "entry_point": entry_point, "inputs": encoded_inputs},
                             ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("execution payload exceeds 16 MiB")
        name = "rsi-audit-" + uuid.uuid4().hex
        started = time.monotonic()
        try:
            try:
                returncode, raw, _stderr = self._run_container(self._build_command(name), payload)
            except _Deadline:
                result = {"status": "timeout", "outputs": [], "cpu_seconds": None}
            except _OutputLimit:
                result = {"status": "protocol_error", "outputs": [], "cpu_seconds": None}
            else:
                if returncode:
                    # A confirmed cgroup OOM is a candidate resource failure.
                    # CLI/startup errors and unexplained worker exits abort.
                    if returncode in (125, 126, 127):
                        raise RuntimeError("Docker could not launch the worker")
                    state_raw = self._control(["inspect", "--format", "{{json .State}}", name])
                    try:
                        state = wire.strict_json_loads(state_raw)
                    except wire.WireError as exc:
                        raise RuntimeError("Docker returned invalid container state") from exc
                    if type(state) is dict and state.get("OOMKilled") is True:
                        result = {"status": "candidate_error", "outputs": [], "cpu_seconds": None}
                    else:
                        raise RuntimeError("worker exited unexpectedly; no model score was produced")
                else:
                    result = self._parse_response(raw, len(inputs))
        finally:
            # This is mandatory even after timeout, malformed output, Popen
            # failure or Ctrl-C. A failed forced cleanup aborts the experiment.
            self._cleanup(name)
        result["wall_seconds"] = time.monotonic() - started
        return result
