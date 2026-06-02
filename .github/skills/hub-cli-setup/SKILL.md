---
name: hub-cli-setup
description: Use when setting up Hugging Face Hub CLI authentication or the recommended GitHub CLI workflow for IsaacLab-Imitation datasets, pushes, PRs, and CI checks.
---

# Hub CLI Setup

Use the repo-owned Pixi environments. Hugging Face CLI support lives in the
`lerobot` environment, while GitHub CLI support lives in the default
environment.

```bash
# Hugging Face Hub CLI for LeRobot dataset access. Use hf, not huggingface-cli.
pixi run -e lerobot hf auth login
pixi run -e lerobot hf auth whoami

# GitHub CLI is recommended for branch, push, PR, and CI workflows.
pixi run gh auth login
pixi run gh auth setup-git --hostname github.com
pixi run gh auth status

# Optional: only for direct git push/pull to https://huggingface.co.
# This uses Git's plaintext store helper, scoped to Hugging Face only.
git config --global credential.https://huggingface.co.helper store

# If you are already logged in:
pixi run -e lerobot hf auth list
TOKEN_NAME=home-ubuntu
pixi run -e lerobot hf auth switch --token-name "$TOKEN_NAME" --add-to-git-credential

# If you are not logged in yet:
pixi run -e lerobot hf auth login --add-to-git-credential

# Remove the Hugging Face-scoped helper later if you no longer want it.
git config --global --unset credential.https://huggingface.co.helper
```
