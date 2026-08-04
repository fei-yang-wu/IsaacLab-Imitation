# Vendored third-party reference models

`g1/g1_29dof_official.xml` is Unitree's own MuJoCo model, fetched verbatim from
`unitreerobotics/unitree_mujoco` (`unitree_robots/g1/g1_29dof.xml`). It is a
REFERENCE for dynamics comparison, not a runtime asset, and it deliberately does
not replace
`source/isaaclab_imitation/.../g1_description/g1_29dof_rev_1_0.xml`:

* the runtime model Isaac Lab actually spawns is the USD, not either MJCF;
* the vendored MJCF is read by `unitree_joint_order.py` for its actuator name
  order, and this file declares 35 joints against the vendored 30, so swapping
  it in would silently permute the joint order the whole stack is pinned to --
  the exact class of failure `wiki/sim2sim-backend-verification.md` documents.

What makes it worth having is that it carries Unitree's own validated passive
parameters, which the vendored MJCF omits entirely:

| parameter | official | vendored |
|---|---|---|
| `armature` | 0.01 | absent |
| `damping` | 0.05 | absent |
| `frictionloss` | 0.2 (0.1 wrists) | absent |

Those are the knobs `newton_mjwarp` exposes, so they are the candidate
explanation for a Newton-vs-PhysX dynamics gap. Drive the comparison with
`scripts/bench/mujoco_reference_tracking_baseline.py --mjcf <this file>`.

## Meshes

`g1/meshes` is a relative symlink into this repo's own
`g1_description/meshes`, so nothing is vendored twice. It is deliberately
*incomplete* for the non-rev files: those reference `waist_yaw_link.STL`,
`waist_roll_link.STL`, `torso_link.STL` and `waist_support_link.STL`, while this
repo ships only the `_rev_1_0` variants. `g1_29dof.xml` and
`g1_29dof_official.xml` therefore fail to compile with
`Error opening file 'meshes/waist_yaw_link.STL'`.

That is not a packaging bug -- it is the revision difference. Use
`g1_29dof_official_dynamics.xml`, which carries the inertials and the passive
joint parameters with no mesh assets at all, whenever you want to load the
non-rev model. See `wiki/sim2sim-backend-verification.md`, "rev_1_0 is a
different robot from `g1_29dof`".
