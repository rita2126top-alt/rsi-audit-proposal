#!/usr/bin/env python3
"""Container-only worker. Never import or run this on the research host.

The supervisor holds no expected answers and computes no pass/fail scores.
Candidate execution occurs in a forked child. Its ordinary stdout/stderr are
discarded, while the supervisor alone emits the outer response and CPU usage.
All values sent to the host must satisfy the non-executable wire grammar.
"""

import inspect
import json
import keyword
import os
import signal
import sys

# -I excludes the script directory from Python's import path. This is a fixed,
# read-only directory inside the image; it is not an attacker-controlled mount.
sys.path.insert(0, "/opt/rsi_audit")
import wire

MAX_BYTES = 16 * 1024 * 1024
MAX_CODE_BYTES = 256 * 1024
NON_NONE_ENTRIES = {"check_str", "text_match_three", "text_starta_endb"}


def _serialize(message):
    raw = json.dumps(message, ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("ascii")
    if len(raw) > MAX_BYTES:
        raise wire.WireError("response exceeds byte limit")
    return raw


def _write_all(fd, raw):
    pos = 0
    while pos < len(raw):
        pos += os.write(fd, raw[pos:pos + 65536])


def _calls(inputs):
    if type(inputs) is not list or not 1 <= len(inputs) <= 4096:
        raise wire.WireError("invalid input count")
    calls = []
    for record in inputs:
        if type(record) is list:
            args, kwargs = record, {}
        elif type(record) is dict and "args" in record and set(record) <= {"args", "kwargs"}:
            args, kwargs = record["args"], record.get("kwargs", {})
        else:
            raise wire.WireError("invalid call record")
        if type(args) is not list or type(kwargs) is not dict or len(args) + len(kwargs) > 128:
            raise wire.WireError("invalid arguments")
        if any(type(k) is not str for k in kwargs):
            raise wire.WireError("invalid keyword")
        calls.append((args, kwargs))
    return calls


def _child(write_fd, code, entry, calls):
    # These callables are retained before exec. Candidate mutation of Python
    # module globals still cannot change the supervisor's separate process.
    exit_process, write = os._exit, os.write
    encode, serialize = wire.encode, _serialize
    try:
        null = os.open("/dev/null", os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        if null > 2 and null != write_fd:
            os.close(null)
        outputs = []
        status = "ok"
        try:
            namespace = {"__name__": "candidate"}
            # The only exec in this project is in this container-only child.
            exec(compile(code, "<candidate>", "exec"), namespace, namespace)
            function = namespace[entry]
            if not callable(function):
                raise TypeError("entry point is not callable")
            signature = inspect.signature(function)
            for args, kwargs in calls:
                signature.bind(*args, **kwargs)
                value = function(*args, **kwargs)
                # Preserve ordinary values for public examples. Only opaque
                # results of the three regex oracles need the non-None marker;
                # arbitrary classes are never reconstructed on the host.
                try:
                    output = encode(value, max_nodes=1_000_000, max_bytes=MAX_BYTES)
                except wire.WireError:
                    if entry in NON_NONE_ENTRIES and value is not None:
                        output = {"t": "non_none"}
                    else:
                        raise
                outputs.append(output)
        except wire.WireError:
            status = "protocol_error"
        except BaseException:
            status = "candidate_error"
        try:
            raw = serialize({"status": status, "outputs": outputs})
        except BaseException:
            raw = b'{"status":"protocol_error","outputs":[]}'
        pos = 0
        while pos < len(raw):
            pos += write(write_fd, raw[pos:pos + 65536])
    except BaseException:
        pass
    finally:
        exit_process(0)


def _run(code, entry, calls):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        _child(write_fd, code, entry, calls)
    os.close(write_fd)
    raw = bytearray()
    oversized = False
    try:
        while True:
            part = os.read(read_fd, 65536)
            if not part:
                break
            if len(raw) + len(part) > MAX_BYTES:
                oversized = True
                os.kill(pid, signal.SIGKILL)
                break
            raw.extend(part)
    finally:
        os.close(read_fd)
    _, wait_status, usage = os.wait4(pid, 0)
    cpu = usage.ru_utime + usage.ru_stime
    empty = {"status": "candidate_error", "outputs": [], "cpu_seconds": cpu}
    if oversized:
        empty["status"] = "protocol_error"
        return empty
    if not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0 or not raw:
        return empty
    try:
        result = wire.strict_json_loads(raw.decode("utf-8"))
        if (type(result) is not dict or set(result) != {"status", "outputs"} or
                result["status"] not in ("ok", "candidate_error", "protocol_error") or
                type(result["outputs"]) is not list or len(result["outputs"]) > len(calls) or
                (result["status"] == "ok" and len(result["outputs"]) != len(calls))):
            raise wire.WireError("invalid candidate response")
        # Decode validates shapes only; no candidate objects can be created.
        for output in result["outputs"]:
            wire.decode(output, max_nodes=1_000_000, max_bytes=MAX_BYTES)
        result["cpu_seconds"] = cpu
        return result
    except (wire.WireError, ValueError, TypeError, UnicodeError, KeyError):
        empty["status"] = "protocol_error"
        return empty


def main():
    result = {"status": "protocol_error", "outputs": [], "cpu_seconds": None}
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise wire.WireError("oversized request")
        request = wire.strict_json_loads(raw.decode("utf-8"))
        if type(request) is not dict or set(request) != {"code", "entry_point", "inputs"}:
            raise wire.WireError("only code, entry point and inputs may enter worker")
        code, entry = request["code"], request["entry_point"]
        if type(code) is not str or not code.strip() or len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise wire.WireError("invalid code size")
        if (type(entry) is not str or not entry.isidentifier() or keyword.iskeyword(entry) or
                entry.startswith("__") or len(entry) > 128):
            raise wire.WireError("invalid entry point")
        inputs = wire.decode(request["inputs"], max_nodes=1_000_000, max_bytes=MAX_BYTES)
        calls = _calls(inputs)
    except (wire.WireError, ValueError, TypeError, UnicodeError, KeyError):
        calls = None
    if calls is not None:
        try:
            result = _run(code, entry, calls)
        except BaseException:
            # Fork/pipe/supervisor failures are infrastructure failures. Exit
            # nonzero so the host aborts, rather than charging a model failure.
            sys.stderr.write("worker supervisor failed\n")
            return 70
    sys.stdout.buffer.write(_serialize(result))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
