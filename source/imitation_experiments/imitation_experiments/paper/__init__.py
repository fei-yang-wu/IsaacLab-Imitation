"""Shared implementation behind the release-facing paper entrypoints.

``experiments/paper/`` keeps the entrypoints a collaborator actually runs --
``run.sh``, the submit and aggregate scripts, the release-bundle builder. What
they are built out of lives here, because it is shared code with tests, and
because a module reachable only by inserting ``experiments/paper`` on
``sys.path`` is not importable from a test, a notebook, or a sibling script
without repeating that hack.

- :mod:`~imitation_experiments.paper.common`: pipeline plumbing (provenance
  records, Hydra override construction, guarded output roots, subprocess
  running).
- :mod:`~imitation_experiments.paper.specs`: the frozen interface and
  latent-mode specifications the paper rows are defined by.
"""

from __future__ import annotations

__all__ = ["common", "specs"]
