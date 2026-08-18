"""CLI entry: ``python -m imitation_experiments.pipeline.cluster <verb> ...``.

Exit codes: 0 success, 2 gate rejection (validation, preflight, confirmation,
drift), 1 unexpected failure.
"""

from __future__ import annotations

import argparse
import sys

from imitation_experiments.paper.common import PipelineError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m imitation_experiments.pipeline.cluster",
        description="Cluster-submission control plane: plan, submit, status, logs, cancel.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    plan = sub.add_parser("plan", help="resolve + preflight + freeze; never submits")
    plan.add_argument("--campaign", required=True, help="path to campaign.yaml")
    plan.add_argument("--arm", required=True)
    plan.add_argument("--seed", required=True, type=int)
    plan.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist override, repeatable (e.g. vars.frame_cap=2000000000)",
    )
    plan.add_argument(
        "--profile", default=None, help="profile name or YAML path override"
    )
    plan.add_argument(
        "--only-stage", default=None, help="plan a single stage of the arm"
    )
    plan.add_argument(
        "--skip-preflight", action="store_true", help="offline plan, no ssh"
    )
    plan.add_argument(
        "--out-root", default=None, help="default: <repo>/logs/cluster_control"
    )

    submit = sub.add_parser("submit", help="execute a confirmed plan")
    submit.add_argument(
        "--plan", required=True, help="plan directory or plan.json path"
    )
    submit.add_argument("--confirm", required=True, metavar="PLAN_SHA")
    submit.add_argument("--allow-drift", action="store_true")
    submit.add_argument("--allow-resubmit", action="store_true")

    status = sub.add_parser("status", help="squeue/sacct view of a submission")
    logs = sub.add_parser("logs", help="tail a submitted stage's Slurm log")
    cancel = sub.add_parser("cancel", help="scancel a submission's jobs")
    for verb in (status, logs, cancel):
        verb.add_argument(
            "--submission", default=None, help="submission json or plan dir"
        )
        verb.add_argument(
            "--campaign", default=None, help="filter when auto-discovering"
        )
    logs.add_argument("--stage", default=None)
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("-n", "--lines", default=100, type=int)
    cancel.add_argument("--stage", default=None)
    cancel.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Imports deferred so `--help` stays fast and verbs stay independent.
    if args.verb == "plan":
        from .plan_cmd import cmd_plan as command
    elif args.verb == "submit":
        from .submit_cmd import cmd_submit as command
    elif args.verb == "status":
        from .status_cmd import cmd_status as command
    elif args.verb == "logs":
        from .status_cmd import cmd_logs as command
    else:
        from .status_cmd import cmd_cancel as command
    try:
        return command(args)
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
