"""Strict JSON defaults for the training CLI, with explicit CLI overrides."""
import argparse
import hashlib
import json
import math
from pathlib import Path


BOOL_FIELDS = {"use_elo", "no_show", "skip_test", "quiet", "augment", "print_config"}
INT_FIELDS = {"history", "epochs", "batch_size", "channels", "blocks", "workers", "seed",
              "convs_per_block", "kernel_size"}
FLOAT_FIELDS = {"lr", "huber_delta", "dropout", "weight_decay", "momentum"}
CHOICES = {"tasks": {"outcome", "all"}, "device": {"auto", "cpu", "cuda", "mps"},
           "select_by": {"outcome", "total"}, "optimizer": {"adamw", "sgd"}}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_values(values, *, allowed=None):
    """Validate JSON types before argparse, then ranges after CLI resolution."""
    if allowed is not None:
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError("unknown configuration keys: " + ", ".join(unknown))
    for key, value in values.items():
        if key in BOOL_FIELDS:
            if type(value) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        elif key in INT_FIELDS:
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif key in FLOAT_FIELDS:
            if not _number(value):
                raise ValueError(f"{key} must be a finite number")
        elif key == "task_weights":
            if (not isinstance(value, list) or len(value) != 5
                    or not all(_number(v) and v >= 0 for v in value) or not any(value)):
                raise ValueError("task_weights must be five finite nonnegative numbers, with at least one positive")
        elif key not in {"config", "config_sha256"} and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if key in CHOICES and value not in CHOICES[key]:
            raise ValueError(f"{key} must be one of {', '.join(sorted(CHOICES[key]))}")
    for key in ("epochs", "batch_size", "channels"):
        if key in values and values[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("blocks", "workers", "history"):
        if key in values and values[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if "seed" in values and not 0 <= values["seed"] < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")
    for key in ("lr", "huber_delta"):
        if key in values and values[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if "weight_decay" in values and values["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    if "dropout" in values and not 0 <= values["dropout"] <= 1:
        raise ValueError("dropout must be in [0, 1]")
    if "momentum" in values and not 0 <= values["momentum"] < 1:
        raise ValueError("momentum must be in [0, 1)")
    if "convs_per_block" in values and values["convs_per_block"] not in (2, 3):
        raise ValueError("convs_per_block must be 2 or 3")
    if "kernel_size" in values and (values["kernel_size"] < 1 or values["kernel_size"] % 2 == 0):
        raise ValueError("kernel_size must be a positive odd integer")


def parse_with_config(parser, argv=None):
    """Resolve built-in defaults < JSON values < explicit command-line options."""
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args(argv)
    config_hash = None
    if known.config:
        path = Path(known.config).expanduser()
        try:
            raw = path.read_bytes()
            loaded = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(loaded, dict):
                raise ValueError("configuration must be a JSON object")
            allowed = {action.dest for action in parser._actions} - {"help", "config", "print_config"}
            validate_values(loaded, allowed=allowed)
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(f"cannot load --config {path}: {error}")
        parser.set_defaults(**loaded)
        config_hash = hashlib.sha256(raw).hexdigest()
    resolved = parser.parse_args(argv)
    try:
        validate_values(vars(resolved))
    except ValueError as error:
        parser.error(str(error))
    if resolved.config:
        resolved.config = str(Path(resolved.config).expanduser().resolve())
    resolved.config_sha256 = config_hash
    return resolved
