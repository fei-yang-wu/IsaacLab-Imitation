"""Bootstrap the split Isaac Sim / CU130 Python runtime before user imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path


site_packages = Path(os.environ["ISAACLAB_CU130_SITE_PACKAGES"]).resolve()
if not (site_packages / "torch").is_dir():
    raise RuntimeError(f"CU130 Torch package is missing under {site_packages}")
site_text = str(site_packages)
if site_text not in sys.path:
    sys.path.append(site_text)

import torch  # noqa: E402

torch_file = Path(torch.__file__).resolve()
if site_packages not in torch_file.parents:
    raise RuntimeError(
        f"Torch resolved from {torch_file}, expected runtime under {site_packages}."
    )
if torch.__version__.split("+", 1)[0] != "2.11.0":
    raise RuntimeError(f"Expected PyTorch 2.11.0, found {torch.__version__}.")
if torch.version.cuda != "13.0":
    raise RuntimeError(f"Expected CUDA 13.0 Torch, found {torch.version.cuda!r}.")
print(
    "[INFO] Verified direct CU130 Kit bridge: "
    f"torch={torch.__version__}, cuda={torch.version.cuda}, origin={torch_file}",
    flush=True,
)
