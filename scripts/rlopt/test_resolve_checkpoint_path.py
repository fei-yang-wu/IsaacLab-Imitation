"""Directory resolution for ``--checkpoint`` in multi-segment resumed runs.

`train_impl` imports Isaac Lab at module scope, so the helper is loaded from
source here instead of importing the module.
"""

from __future__ import annotations

import ast
import os
import textwrap
from pathlib import Path

import pytest

_SOURCE = Path(__file__).with_name("train_impl.py")


def _load_helper():
    tree = ast.parse(_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_checkpoint_path":
            namespace: dict = {}
            exec(  # noqa: S102 - the source is this repo's own file
                "import os\nimport re\n" + textwrap.dedent(ast.unparse(node)),
                namespace,
            )
            return namespace["_resolve_checkpoint_path"]
    raise AssertionError("_resolve_checkpoint_path not found in train_impl.py")


resolve = _load_helper()


def test_a_file_path_passes_through(tmp_path: Path) -> None:
    ckpt = tmp_path / "model_step_500.pt"
    ckpt.write_bytes(b"")
    assert resolve(str(ckpt)) == str(ckpt)


def test_a_missing_file_path_still_passes_through(tmp_path: Path) -> None:
    # Loud failure belongs to torch.load, which reports the path it could not
    # open; the resolver must not turn an explicit file into a directory scan.
    missing = tmp_path / "model_step_500.pt"
    assert resolve(str(missing)) == str(missing)


def test_a_directory_resolves_to_the_newest_checkpoint(tmp_path: Path) -> None:
    for index, step in enumerate((500_000_000, 2_000_000_000, 1_500_000_000)):
        ckpt = tmp_path / f"model_step_{step}.pt"
        ckpt.write_bytes(b"")
        os.utime(ckpt, (1_000 + index, 1_000 + index))
    # 1_500_000_000 is newest but NOT the highest step: selection is by mtime.
    assert resolve(str(tmp_path)) == str(tmp_path / "model_step_1500000000.pt")


def test_it_walks_the_per_run_logger_subdirectories(tmp_path: Path) -> None:
    # The real layout: <tracker>/rlopt_train/<timestamp>_wandb-<id>/model_step_N.pt
    first = tmp_path / "rlopt_train" / "2026-08-15_10-00-00_wandb-aaa"
    second = tmp_path / "rlopt_train" / "2026-08-15_22-00-00_wandb-bbb"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    old_ckpt = first / "model_step_2500000000.pt"
    new_ckpt = second / "model_step_500000000.pt"
    old_ckpt.write_bytes(b"")
    new_ckpt.write_bytes(b"")
    os.utime(old_ckpt, (1_000, 1_000))
    os.utime(new_ckpt, (2_000, 2_000))
    # Segment 2's step counter restarted at zero, so the highest-numbered file
    # in the tree is segment 1's. Resuming by step would walk backwards.
    assert resolve(str(tmp_path)) == str(new_ckpt)


def test_unrelated_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "model_step_10.pt").write_bytes(b"")
    (tmp_path / "latest.pt").write_bytes(b"")
    (tmp_path / "model_step_20.pt.tmp").write_bytes(b"")
    (tmp_path / "model_step_abc.pt").write_bytes(b"")
    assert resolve(str(tmp_path)) == str(tmp_path / "model_step_10.pt")


def test_an_empty_tree_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "rlopt_train" / "run").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no model_step"):
        resolve(str(tmp_path))
