# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages
from isaaclab_imitation import newton_wrench_frame as _newton_wrench_frame

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

# Isaac Lab 3.0.0b2.post1 writes body-frame external wrenches into Newton's
# world-frame body_f. Every task here that applies a wrench (floating hands,
# virtual objects, push events) needs the frame corrected before the write.
_newton_wrench_frame.install()
