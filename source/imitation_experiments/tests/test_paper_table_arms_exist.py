"""Every arm the paper tables name must exist in the campaign and be labelled.

The tables and the convergence figures are built from two different places --
`experiments/paper/build_ablation_tables.py` holds the LaTeX row structure,
while the figures read `paper_label` out of the campaign. A rename or a typo in
either would silently produce a `--` row or an arm-id legend entry, so this
pins the two together.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from imitation_experiments.paths import REPO_ROOT

CAMPAIGN = REPO_ROOT / "experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml"
BUILDER = REPO_ROOT / "experiments/paper/build_ablation_tables.py"

pytestmark = pytest.mark.skipif(
    not CAMPAIGN.is_file() or not BUILDER.is_file(),
    reason="star-v2 campaign or table builder not present",
)


def _builder():
    spec = importlib.util.spec_from_file_location("build_ablation_tables", BUILDER)
    module = importlib.util.module_from_spec(spec)
    # `dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so the module must be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign_arms() -> dict:
    return yaml.safe_load(Path(CAMPAIGN).read_text())["arms"]


def test_every_table_row_names_a_real_arm():
    arms = _campaign_arms()
    unknown = {
        row.arm
        for table in _builder().TABLES.values()
        for row in table.body
        if row is not None and row.arm and row.arm not in arms
    }
    assert not unknown, f"table rows name arms absent from the campaign: {unknown}"


def test_every_arm_carries_a_paper_label_for_the_figures():
    missing = [
        name
        for name, spec in _campaign_arms().items()
        if not str((spec.get("vars") or {}).get("paper_label", "")).strip()
    ]
    assert not missing, f"arms without paper_label: {missing}"


def test_every_arm_carries_a_section_so_it_reaches_a_figure():
    missing = [
        name
        for name, spec in _campaign_arms().items()
        if not int((spec.get("vars") or {}).get("section", 0))
    ]
    assert not missing, f"arms without a section: {missing}"


def test_paper_labels_are_unique():
    labels: dict[str, str] = {}
    for name, spec in _campaign_arms().items():
        label = str((spec.get("vars") or {})["paper_label"])
        assert label not in labels, (
            f"{name} and {labels[label]} share the label {label!r}; "
            "a figure legend could not tell them apart"
        )
        labels[label] = name
