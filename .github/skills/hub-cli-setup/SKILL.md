---
name: hub-cli-setup
description: Use when setting up Hugging Face Hub CLI authentication or the recommended GitHub CLI workflow for IsaacLab-Imitation datasets, pushes, PRs, and CI checks.
---

# Hub CLI Setup

Ask the user which conda environment they prefer if the current thread has not
already specified one. `SL` and `SkillLearning` are examples only.

```bash
CONDA_ENV="${CONDA_ENV:-SL}"
conda activate "$CONDA_ENV"

# Hugging Face Hub CLI for LeRobot dataset access. Use hf, not huggingface-cli.
uv pip install --system -U "huggingface_hub[cli]"
hf auth login
hf auth whoami

# GitHub CLI is recommended for branch, push, PR, and CI workflows.
conda install -y -c conda-forge gh
gh auth login
gh auth setup-git --hostname github.com
gh auth status

# Optional: only for direct git push/pull to https://huggingface.co.
# This uses Git's plaintext store helper, scoped to Hugging Face only.
git config --global credential.https://huggingface.co.helper store

# If you are already logged in:
hf auth list
TOKEN_NAME=home-ubuntu
hf auth switch --token-name "$TOKEN_NAME" --add-to-git-credential

# If you are not logged in yet:
hf auth login --add-to-git-credential

# Remove the Hugging Face-scoped helper later if you no longer want it.
git config --global --unset credential.https://huggingface.co.helper
```
