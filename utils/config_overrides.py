"""
Utilities to override nested configuration values via CLI using dot notation.

Example:
  python script.py --config_file cfg.yml --Test.save_dir /out --Output.save_to_disk=false

Supports:
  - --Key=value
  - --Key value
  - Boolean flags: --Key (interpreted as true)
  - YAML parsing of values for correct types (ints, floats, lists, bools)
"""
from typing import Any, Dict, List, Tuple

import yaml


def parse_cli_overrides(unknown_args: List[str]) -> List[Tuple[str, str]]:
    overrides: List[Tuple[str, str]] = []
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not isinstance(token, str) or not token.startswith("--"):
            i += 1
            continue
        keyval = token[2:]
        # Support --Key=value
        if "=" in keyval:
            k, v = keyval.split("=", 1)
            overrides.append((k, v))
            i += 1
            continue
        # Support --Key value and boolean flags defaulting to true
        if i + 1 < len(unknown_args) and isinstance(unknown_args[i + 1], str) and not unknown_args[i + 1].startswith("--"):
            v = unknown_args[i + 1]
            i += 2
        else:
            v = "true"
            i += 1
        overrides.append((keyval, v))
    return overrides


def _set_in_config(cfg: Dict[str, Any], dotted_key: str, value_str: str) -> None:
    keys = dotted_key.split(".")
    d: Dict[str, Any] = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    try:
        parsed_val = yaml.safe_load(value_str)
    except Exception:
        parsed_val = value_str
    d[keys[-1]] = parsed_val


def apply_cli_overrides_from_unknown_args(config: Dict[str, Any], unknown_args: List[str]) -> None:
    """Apply dotted key overrides found in unknown_args to config in-place."""
    for k, v in parse_cli_overrides(unknown_args):
        if "." in k:  # avoid clashing with normal flags
            _set_in_config(config, k, v)
