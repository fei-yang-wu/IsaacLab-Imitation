# Vega U plus Wuji Hand asset

This directory contains the robot-only Vega U plus dual Wuji Hand V2 Beta 1
MJCF and every mesh referenced by that MJCF. The files are tracked with Git
LFS where they are binary mesh assets.

Source provenance:

- source host: `mel07876d`
- source tree: `/mnt/hsstorage/fwu91/Projects/DexManip/simulations/wuji_vega_u_grasp/build`
- source model SHA-256: `23f3aed6c2d8099000ac98ee444b73868c02f899447d000d7c4c0a70cbab034c`
- Embodied-Control integration: `dev-dex`, commit `54dc72d4225d16a039052b5847fd968fbc363905`

The packaged model is robot-only. The table and cube scene are intentionally
not included. The Isaac task uses this MJCF by default and converts it lazily
to USD through `MjcfFileCfg`.

The task wrist frame is the palm site frame. `right_palm` is an identity site
on `r_mount`, and `left_palm` is an identity site on `l_mount`. Newton exposes
the two mount bodies for Jacobian control and live state. Do not use the
`R_ee` and `L_ee` parent frames as Reference wrists; each parent differs from
its mount by a fixed pi rotation.

After a fresh clone, materialize binary files with:

```bash
git lfs pull
```

Run the contract audit from the repository root:

```bash
pixi run python scripts/data/validate_vega_wuji_asset.py \
  --model source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji/vega_u_wuji_v2_beta1_with_mount.xml
```
