#!/usr/bin/env python3
"""Build OpenForesight train/val parquets in an isolated process.

Embedding / matcher work can allocate GPU memory. Running this as a subprocess that exits
releases the CUDA context before the SkyRL driver starts (``execve`` alone keeps the same PID
and can leave VRAM pinned, breaking ``ray.init()`` right after ``Started a local Ray instance``).
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_run_skyrl_module():
    mod_path = REPO_ROOT / "scripts" / "run_skyrl_openforesight_search.py"
    mod = types.ModuleType("_fsim_run_skyrl_prepare")
    mod.__file__ = str(mod_path)
    # Load the launcher from source so parquet prep never picks up a stale pyc on shared storage.
    code = compile(mod_path.read_text(encoding="utf-8"), str(mod_path), "exec")
    exec(code, mod.__dict__)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Expanded YAML config path")
    p.add_argument("--run-name", required=True)
    p.add_argument("--paths-out", required=True, help="Write JSON with train, val, prepared_root")
    args = p.parse_args()

    rs = _load_run_skyrl_module()
    from pathing import expand_env_tree, raise_for_unresolved_env_vars

    config_path = Path(args.config).resolve()
    config = expand_env_tree(rs._read_yaml(config_path))
    if hasattr(rs, "_apply_reserved_aux_cuda_visible_devices"):
        config = rs._apply_reserved_aux_cuda_visible_devices(config)
    raise_for_unresolved_env_vars(config, f"SkyRL parquet prep {config_path}")
    train_path, val_path, prepared_root = rs._prepare_openforesight_parquets(config, run_name=args.run_name)
    out = Path(args.paths_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"train": train_path, "val": val_path, "prepared_root": str(prepared_root)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
