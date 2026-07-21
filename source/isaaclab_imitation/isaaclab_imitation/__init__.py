# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# UI extensions require a running Omniverse Kit process. The package must stay
# importable for strict kit-less Newton training.
import sys as _sys

if "omni.kit.app" in _sys.modules:
    from .ui_extension_example import *
