"""Make the stable paper entrypoints importable from this campaign's tests.

``experiments/paper/`` keeps only the guarded, release-facing entrypoints, while
the shared implementation and its tests live here. A few tests exercise both
sides of that boundary, so put the paper directory on ``sys.path`` alongside the
sibling modules pytest already inserts.
"""

from __future__ import annotations

import sys

from _repo_paths import PAPER_DIR

if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))
