# Unitree G1 Assets

This directory vendors the Unitree G1 29-DoF URDF, MuJoCo XML, source license,
and mesh files referenced by those robot descriptions so IsaacLab-Imitation can
run G1 tasks without importing `unitree_rl_lab`.

The local source checkout did not contain a generated USD for this robot. Isaac
Lab imports the packaged URDF directly and can generate its own converted USD
cache at runtime.
