"""Bounded, deterministic JSON values for the untrusted-code boundary.

No pickle, object hooks, import paths, constructors, or executable expressions
cross this boundary. Exact built-in types are used so candidate-defined
``__iter__``, ``__repr__`` and equality methods never run on the host. A task
bundle can contain sets, tuples and complex numbers without losing their type.
The executor applies a smaller, independent byte limit to each request/reply.
"""

from __future__ import annotations

import json
import math

MAX_DEPTH = 64
MAX_NODES = 4_000_000
MAX_BYTES = 256 * 1024 * 1024
MAX_STRING_BYTES = 2 * 1024 * 1024
MAX_INTEGER_BITS = 16_384


class WireError(ValueError):
    """A value is outside the explicitly supported, bounded wire language."""


class _NonNone:
    __slots__ = ()


# Only the three MBPP non-None oracles may interpret this marker. It is never
# treated as equality to an ordinary value, nor as a general successful result.
NON_NONE = _NonNone()


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _byte_check(value, maximum):
    try:
        length = len(_json(value).encode("ascii"))
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise WireError("wire value is not strict JSON") from exc
    if length > maximum:
        raise WireError("wire value exceeds byte limit")


def encode(value, *, max_nodes=MAX_NODES, max_depth=MAX_DEPTH,
           max_bytes=MAX_BYTES):
    """Encode a supported Python value, with deterministic set/dict ordering.

Non-finite floats use an explicit tag; nonstandard JSON NaN/Infinity is never
emitted. Decimal integer strings preserve integers exactly across JSON readers.
Cycles, excessively large values and arbitrary classes are rejected.
"""
    count = 0
    active = set()

    def visit(v, depth):
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise WireError("wire value exceeds structural limit")
        typ = type(v)
        if v is NON_NONE:
            return {"t": "non_none"}
        if v is None:
            return {"t": "none"}
        if typ is bool:
            return {"t": "bool", "v": v}
        if typ is int:
            if v.bit_length() > MAX_INTEGER_BITS:
                raise WireError("integer exceeds bit limit")
            try:
                return {"t": "int", "v": str(v)}
            except ValueError as exc:
                raise WireError("integer exceeds decimal conversion limit") from exc
        if typ is float:
            if not math.isfinite(v):
                return {"t": "nonfinite", "v": "nan" if math.isnan(v)
                        else ("+inf" if v > 0 else "-inf")}
            return {"t": "float", "v": v}
        if typ is str:
            if len(v.encode("utf-8", errors="surrogatepass")) > MAX_STRING_BYTES:
                raise WireError("string exceeds byte limit")
            return {"t": "str", "v": v}
        if typ is complex:
            return {"t": "complex", "v": [visit(v.real, depth + 1),
                                             visit(v.imag, depth + 1)]}
        if typ not in (list, tuple, set, frozenset, dict):
            raise WireError("unsupported Python type: " + typ.__name__[:80])
        ident = id(v)
        if ident in active:
            raise WireError("cyclic values are unsupported")
        active.add(ident)
        try:
            if typ is dict:
                items = [[visit(k, depth + 1), visit(x, depth + 1)]
                         for k, x in v.items()]
                items.sort(key=lambda pair: _json(pair[0]))
                return {"t": "dict", "v": items}
            items = [visit(x, depth + 1) for x in v]
            if typ in (set, frozenset):
                items.sort(key=_json)
            return {"t": typ.__name__, "v": items}
        finally:
            active.remove(ident)

    try:
        encoded = visit(value, 0)
    except RecursionError as exc:
        raise WireError("wire nesting exceeds Python recursion limit") from exc
    _byte_check(encoded, max_bytes)
    return encoded


def decode(encoded, *, max_nodes=MAX_NODES, max_depth=MAX_DEPTH,
           max_bytes=MAX_BYTES):
    """Decode only the finite, allowlisted wire grammar; reject extra fields."""
    _byte_check(encoded, max_bytes)
    count = 0

    def visit(v, depth):
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise WireError("wire value exceeds structural limit")
        if type(v) is not dict or type(v.get("t")) is not str:
            raise WireError("each wire value needs a type tag")
        tag = v["t"]
        fields = {"t"} if tag in ("none", "non_none") else {"t", "v"}
        if set(v) != fields:
            raise WireError("invalid fields for wire type")
        if tag == "none":
            return None
        if tag == "non_none":
            return NON_NONE
        data = v.get("v")
        if tag == "bool" and type(data) is bool:
            return data
        if tag == "int" and type(data) is str:
            # A small lexical bound also protects int() before bit_length().
            if not data or len(data) > 4300:
                raise WireError("invalid integer length")
            digits = data[1:] if data.startswith("-") else data
            if not digits.isascii() or not digits.isdigit():
                raise WireError("invalid decimal integer")
            if len(digits) > 1 and digits[0] == "0" or data == "-0":
                raise WireError("noncanonical decimal integer")
            try:
                result = int(data)
            except ValueError as exc:
                raise WireError("invalid decimal integer") from exc
            if result.bit_length() > MAX_INTEGER_BITS:
                raise WireError("integer exceeds bit limit")
            return result
        if tag == "float" and type(data) in (int, float):
            try:
                result = float(data)
            except (ValueError, OverflowError) as exc:
                raise WireError("invalid float") from exc
            if not math.isfinite(result):
                raise WireError("nonfinite float requires explicit tag")
            return result
        if tag == "nonfinite" and type(data) is str:
            if data not in ("nan", "+inf", "-inf"):
                raise WireError("invalid nonfinite tag")
            return {"nan": float("nan"), "+inf": float("inf"),
                    "-inf": -float("inf")}[data]
        if tag == "str" and type(data) is str:
            if len(data.encode("utf-8", errors="surrogatepass")) > MAX_STRING_BYTES:
                raise WireError("string exceeds byte limit")
            return data
        if tag == "complex" and type(data) is list and len(data) == 2:
            real, imag = [visit(x, depth + 1) for x in data]
            if type(real) not in (int, float) or type(imag) not in (int, float):
                raise WireError("complex components must be real numbers")
            return complex(real, imag)
        if tag in ("list", "tuple", "set", "frozenset") and type(data) is list:
            items = [visit(x, depth + 1) for x in data]
            if tag == "list":
                return items
            if tag == "tuple":
                return tuple(items)
            try:
                result = set(items)
            except TypeError as exc:
                raise WireError("unhashable set item") from exc
            if len(result) != len(items):
                raise WireError("duplicate serialized set items")
            return result if tag == "set" else frozenset(result)
        if tag == "dict" and type(data) is list:
            result = {}
            for item in data:
                if type(item) is not list or len(item) != 2:
                    raise WireError("dict entry must be a key/value pair")
                key, val = visit(item[0], depth + 1), visit(item[1], depth + 1)
                try:
                    if key in result:
                        raise WireError("duplicate serialized dictionary key")
                    result[key] = val
                except TypeError as exc:
                    raise WireError("unhashable dictionary key") from exc
            return result
        raise WireError("invalid value or unknown wire tag: " + tag[:80])

    try:
        return visit(encoded, 0)
    except RecursionError as exc:
        raise WireError("wire nesting exceeds Python recursion limit") from exc


def strict_json_loads(text):
    """Reject NaN literals and duplicate object keys before wire validation."""
    def no_constant(_):
        raise WireError("nonstandard JSON numeric literal")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise WireError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        return json.loads(text, parse_constant=no_constant,
                          object_pairs_hook=unique_pairs)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise WireError("invalid strict JSON") from exc
