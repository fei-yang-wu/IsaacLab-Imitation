"""Populations that mirror the SONIC paper's own evaluation splits.

The SONIC paper (arXiv 2511.07820v3, Science Robotics 11(117)) reports its
tracking numbers on two held-out splits of its motion-capture corpus,
test-content and test-repetition, and gives their main-category composition in
Table 2. It never publishes the clip names, so the splits cannot be rebuilt.
What CAN be rebuilt is a population with the same main-category mixture drawn
from the public part of the same corpus, which is what this module does.

Why the mixture is worth matching at all: it is the only part of the paper's
population that is stated in numbers. Do not expect it to move a score. On the
two stored released-checkpoint blocks, reweighting from the raw block mixture
to either paper mixture moves micro MPJPE-L by 0.1-0.4 mm. The mixture makes
the population *nameable*, so a row can say which paper column it faces.

Read `wiki/sonic-release-checkpoint-tier2.md` for what a released-checkpoint
row on this population may be compared against, and in particular for the
model-size finding: the public `last.pt` carries a six-layer action decoder
(`[2048, 2048, 1024, 1024, 512, 512]`) where the paper's Table S1 specifies
eight layers (`[4096, 4096, 2048, 2048, 1024, 1024, 512, 512]`). The public
checkpoint is the paper's 16M rung, not its 42M flagship, so it reproduces the
paper's Table 4(a) rows and cannot reach its Table 4(c) rows.

Nothing here reads a policy, a rollout, or a score, so a population built with
it cannot be tuned to flatter a checkpoint. It shares that property with
`evaluation.clip_features`, and unlike that module it applies no difficulty
band: the paper's splits are not difficulty-filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Iterable, Mapping, Sequence

__all__ = [
    "BONES_CATEGORY_TO_SONIC_GROUP",
    "CorpusClip",
    "SONIC_DEPLOYMENT_FAMILIES",
    "SONIC_TABLE2_TEST_CONTENT",
    "SONIC_TABLE2_TEST_REPETITION",
    "deployment_family",
    "load_corpus_clips",
    "mixture_shares",
    "select_deployment_ranks",
    "select_proxy_ranks",
    "sonic_group",
]


# Clip counts read verbatim from SONIC Table 2, one entry per main-category row.
# The two rows our corpus cannot supply are kept here with their real counts so
# the omission is visible rather than silently absent; `mixture_shares` drops
# every group the corpus does not carry and renormalizes what is left.
SONIC_TABLE2_TEST_REPETITION: Mapping[str, int] = {
    "Locomotion": 2683,
    "Gestures": 1125,
    "Acting": 20,
    "Combat": 0,
    "Props": 253,
    "Dance": 485,
    "Injured": 528,
    "ActionTool": 322,
    "Others": 890,
}
"""Table 2, test-repetition column: 6,306 clips over 1,088 sub-categories."""

SONIC_TABLE2_TEST_CONTENT: Mapping[str, int] = {
    "Locomotion": 2481,
    "Gestures": 1488,
    "Acting": 0,
    "Combat": 0,
    "Props": 701,
    "Dance": 504,
    "Injured": 1167,
    "ActionTool": 228,
    "Others": 429,
}
"""Table 2, test-content column: 6,998 clips over 182 unseen sub-categories."""


# BONES-SEED ships 21 `category` values where SONIC's Table 2 aggregates into
# nine. This is that aggregation. Two Table 2 groups have no BONES-SEED
# counterpart: "Acting" (68,742 clips in SONIC's train split) and "Combat"
# (50,162) are almost entirely absent from the public release -- the corpus
# holds 16 "Martial Arts" and 4 "Magic" clips against SONIC's 50,162 combat
# clips. Both are also near-empty in the two test splits (20 and 0 clips), so a
# proxy for a TEST split loses little; a proxy for the TRAIN split would be
# meaningless.
BONES_CATEGORY_TO_SONIC_GROUP: Mapping[str, str] = {
    "Basic Locomotion Neutral": "Locomotion",
    "Basic Locomotion Styles": "Locomotion",
    "Advanced Locomotion": "Locomotion",
    "Unusual Locomotion": "Locomotion",
    "Gestures": "Gestures",
    "Communication": "Gestures",
    "Looking and Pointing": "Gestures",
    "Object Manipulation": "Props",
    "Object Interaction": "Props",
    "Dancing": "Dance",
    "Household": "ActionTool",
    "Complex Actions": "ActionTool",
    "Consuming": "ActionTool",
    "Baseline": "Others",
    "Sports": "Others",
    "Other": "Others",
    "Environments": "Others",
    "Stunts": "Others",
    "Magic": "Others",
    "Martial Arts": "Others",
}

# SONIC lists Injured as a main category; BONES-SEED does not, and spreads
# those clips across six `category` values, 88% of them under "Basic Locomotion
# Styles". The name prefix recovers 10,558 clips against SONIC's 11,081 across
# all three splits, so it is the right carve-out. It is applied BEFORE the
# category map, which is why an injured walk counts as Injured, not Locomotion.
_INJURED_NAME = re.compile(r"(?:^|_)(?:inj|injured)")


@dataclass(frozen=True)
class CorpusClip:
    """One clip of a reference-arrays tree, with its BONES-SEED category."""

    rank: int
    motion: str
    category: str


def sonic_group(clip: CorpusClip) -> str:
    """The SONIC Table 2 main-category group one clip belongs to."""
    if _INJURED_NAME.search(clip.motion.lower()):
        return "Injured"
    try:
        return BONES_CATEGORY_TO_SONIC_GROUP[clip.category]
    except KeyError:
        # An unmapped category is a corpus change, not a clip to quietly bin.
        raise KeyError(
            f"clip {clip.motion!r} (rank {clip.rank}) carries BONES-SEED category "
            f"{clip.category!r}, which has no SONIC Table 2 group. Extend "
            "BONES_CATEGORY_TO_SONIC_GROUP."
        ) from None


def load_corpus_clips(
    reference_arrays_dir: str | Path, selection_json: str | Path
) -> list[CorpusClip]:
    """Join a reference-arrays tree's rank order to BONES-SEED categories.

    `reference_arrays_dir` supplies rank -> motion name through
    `reference_arrays_manifest.json`; `selection_json` is the selection written
    by `scripts/data/select_bones_seed_sonic.py`, which carries one `category`
    per motion. The two disagree on one separator: the selection keeps
    BONES-SEED's `move__actor` double underscore and the reference tree
    collapses it, so the join normalizes it away.

    A clip the selection does not describe keeps category `"Other"` rather than
    failing the load: six of the 129,785 clips in the frozen tree are missing
    from the selection, which is a provenance wart, not a reason to refuse.
    """
    manifest = json.loads(
        (Path(reference_arrays_dir) / "reference_arrays_manifest.json").read_text()
    )
    names = [entry[1] for entry in manifest["traj_info"]["ordered_traj_list"]]
    selection = json.loads(Path(selection_json).read_text())["motions"]
    by_name = {
        str(record["filename"]).replace("__", "_"): record for record in selection
    }
    return [
        CorpusClip(
            rank=rank,
            motion=name,
            category=str((by_name.get(name) or {}).get("category", "Other")),
        )
        for rank, name in enumerate(names)
    ]


def mixture_shares(
    mixture: Mapping[str, int], available: Iterable[str]
) -> dict[str, float]:
    """Normalize a Table 2 column over the groups a corpus can actually supply.

    Groups with a zero count, and groups absent from `available`, are dropped
    before normalizing, so the returned shares sum to 1 over what is left.
    """
    present = set(available)
    kept = {
        group: count
        for group, count in mixture.items()
        if count > 0 and group in present
    }
    if not kept:
        raise ValueError("no group of the requested mixture exists in the corpus.")
    total = sum(kept.values())
    return {group: count / total for group, count in kept.items()}


def select_proxy_ranks(
    clips: Sequence[CorpusClip],
    *,
    mixture: Mapping[str, int] = SONIC_TABLE2_TEST_REPETITION,
    count: int = 4096,
    seed: int = 20260825,
) -> list[int]:
    """Ranks of a population whose group mixture matches one Table 2 column.

    Pass clips for the ENTIRE corpus. Each group is sampled without replacement
    at its share of `count`; the largest group absorbs the rounding remainder so
    the board is exactly `count` clips. No difficulty band and no name filter
    beyond the SONIC release keyword list already applied when the corpus was
    selected -- the paper's splits are not difficulty-filtered, and its
    deployability filter IS that keyword list.
    """
    grouped: dict[str, list[int]] = {}
    for clip in clips:
        grouped.setdefault(sonic_group(clip), []).append(clip.rank)

    shares = mixture_shares(mixture, grouped)
    quotas = {group: round(count * share) for group, share in shares.items()}
    largest = max(quotas, key=lambda group: quotas[group])
    quotas[largest] += int(count) - sum(quotas.values())

    rng = random.Random(int(seed))
    chosen: list[int] = []
    for group in sorted(quotas):
        quota = quotas[group]
        population = sorted(grouped[group])
        if quota > len(population):
            raise ValueError(
                f"group {group!r} needs {quota} clips and the corpus holds "
                f"{len(population)}."
            )
        chosen.extend(rng.sample(population, quota))
    return sorted(chosen)


# ---------------------------------------------------------------------------
# The 123-clip deployment-set reconstruction.
#
# READ THIS BEFORE USING THE BOARD BELOW.
#
# SONIC deploys its 42M model on "123 diverse motion sequences" and reports
# 100% success / 22.3 mm MPJPE-L in simulation for that set (Figure 2(k-l)).
# The set is never enumerated. Figure S2's caption names the families it shows
# -- "hip-hop dance, stage bow, high jump, kick, crouch walk, and grovel" --
# and calls them REPRESENTATIVE examples, not the whole list; the project page
# adds squatting, kneeling, hand crawling, elbow crawling and boxing.
#
# `SONIC_DEPLOYMENT_FAMILIES` is those eleven names turned into corpus
# patterns. Hand crawl and elbow crawl merge because BONES-SEED clip names do
# not distinguish them.
#
# What a row on this board IS: our balanced draw from the families SONIC says
# it deployed, at the size it says it deployed, scored on our protocol.
#
# What it is NOT, and must never be written as: SONIC's deployment set, or a
# reproduction of 22.3 mm. The family list is a partial reading of a figure
# caption, the clips are ours, and 123 clips is a small enough board that
# population noise is large. Treat any number from it as a DIAGNOSTIC. The
# comparison board is `bones_testbed4096_v1`; the paper-facing calibration
# board is `sonic_proxy_testrep4096_v1`.
#
# The 2026-08-17 "hardware-plausible" board was deleted for selecting on ease.
# This one differs in that its criterion is the paper's own enumeration rather
# than a kinematic filter, and it keeps squat, kneel, crawl and boxing -- but
# that difference does not make it a comparison board.
SONIC_DEPLOYMENT_FAMILIES: Mapping[str, tuple[str, str | None]] = {
    # family -> (include pattern, exclude pattern or None)
    "hiphop_dance": (r"dance_hiphop", None),
    # `bow_saw_cutting_tree` is tool use, not a stage bow.
    "stage_bow": (r"(?:^|_)bow(?:_|$)", r"bow_saw"),
    "high_jump": (r"high_jump", None),
    "kick": (r"(?:^|_)kick", None),
    "crouch_walk": (r"crouch", None),
    "grovel": (r"grovel", None),
    "squat": (r"squat", None),
    "kneel": (r"kneel", None),
    "crawl": (r"crawl|on_all_fours", None),
    "boxing": (r"shadow_boxing", None),
}

_DEPLOYMENT_COMPILED = {
    family: (re.compile(include), re.compile(exclude) if exclude else None)
    for family, (include, exclude) in SONIC_DEPLOYMENT_FAMILIES.items()
}


def deployment_family(clip: CorpusClip) -> str | None:
    """First deployment family a clip's name matches, or None.

    Families are tried in `SONIC_DEPLOYMENT_FAMILIES` order and a clip counts
    once, so the 138 clips matching two families do not inflate a quota.
    """
    lowered = clip.motion.lower()
    for family, (include, exclude) in _DEPLOYMENT_COMPILED.items():
        if include.search(lowered) and not (exclude and exclude.search(lowered)):
            return family
    return None


def select_deployment_ranks(
    clips: Sequence[CorpusClip], *, count: int = 123, seed: int = 20260825
) -> list[int]:
    """Ranks of a balanced draw across the named deployment families.

    Equal quota per family, because the paper calls the set "diverse" and gives
    no per-family counts. A family smaller than its quota contributes all of
    its clips and the shortfall is redistributed over the families that still
    have headroom, so the board is exactly `count` clips.
    """
    grouped: dict[str, list[int]] = {}
    for clip in clips:
        family = deployment_family(clip)
        if family is not None:
            grouped.setdefault(family, []).append(clip.rank)
    if not grouped:
        raise ValueError("no clip of the corpus matches a deployment family.")

    sizes = {family: len(ranks) for family, ranks in grouped.items()}
    if int(count) > sum(sizes.values()):
        raise ValueError(
            f"deployment families hold {sum(sizes.values())} clips, fewer than "
            f"the requested {count}."
        )

    # Water-fill: raise a common per-family level until one more would overshoot,
    # so a family only falls short of the level when the corpus caps it.
    level = 0
    while sum(min(size, level + 1) for size in sizes.values()) <= int(count):
        level += 1
    quotas = {family: min(size, level) for family, size in sizes.items()}
    # Fewer leftovers than families with headroom, by construction of `level`.
    remaining = int(count) - sum(quotas.values())
    for family in sorted(grouped):
        if remaining == 0:
            break
        if quotas[family] < sizes[family]:
            quotas[family] += 1
            remaining -= 1

    rng = random.Random(int(seed))
    chosen: list[int] = []
    for family in sorted(quotas):
        chosen.extend(rng.sample(sorted(grouped[family]), quotas[family]))
    return sorted(chosen)
