"""Check collector/trainer distribution agreement before any PPO update."""

import json


def check_collector(agent, num_envs, output):
    import torch
    from rlopt.agent.ppo.ppo import RunningMeanStdCatInputs

    def norms(module):
        return [m for m in module.modules() if isinstance(m, RunningMeanStdCatInputs)]

    def describe(module):
        return [
            {"training": m.training, "count": float(m.count)} for m in norms(module)
        ]

    def mismatch(rollout):
        flat = rollout.reshape(-1)
        op = agent.actor_critic.get_policy_operator()
        op.eval()
        vals = []
        with torch.no_grad():
            for batch in flat.split(num_envs):
                context = agent._prepare_kl_context(batch, op)
                vals.append(float(agent._compute_kl_after_update(context, op)))
        return {"mean": sum(vals) / len(vals), "maximum": max(vals)}

    report = {
        "optimizer_updates": 0,
        "original": describe(agent.policy),
        "collector": describe(agent.collector.policy),
        "normalizer_shared": [
            a is b for a, b in zip(norms(agent.policy), norms(agent.collector.policy))
        ],
    }
    run = agent.init_metadata()
    first = agent.collect(run, 0)
    report["after_collect"] = {
        "original": describe(agent.policy),
        "collector": describe(agent.collector.policy),
        "stored_vs_current_kl": mismatch(first.rollout),
    }
    agent.collector.policy.eval()
    agent.collector.update_policy_weights_()
    agent.collector.reset()
    second = agent.collect(run, 1)
    report["after_explicit_eval_and_sync"] = {
        "original": describe(agent.policy),
        "collector": describe(agent.collector.policy),
        "stored_vs_current_kl": mismatch(second.rollout),
    }
    report["loss_normalizer_shared"] = [
        a is b
        for a, b in zip(norms(agent.policy), norms(agent.loss_module.actor_network))
    ]
    agent.config.ppo.normalizer_update_mode = "rollout"
    agent.actor_critic.train()
    agent.adv_module.train()
    with agent._freeze_rollout_normalizer_updates():
        agent.data_buffer.empty()
        agent.pre_iteration_compute(second.rollout)
        batch = next(iter(agent.data_buffer))
        op = agent.actor_critic.get_policy_operator()
        context = agent._prepare_kl_context(batch, op)
        report["before_loss"] = {
            "source": describe(agent.policy),
            "loss": describe(agent.loss_module.actor_network),
        }
        with torch.no_grad():
            agent.loss_module(batch)
        report["after_loss_without_optimizer"] = {
            "source": describe(agent.policy),
            "loss": describe(agent.loss_module.actor_network),
            "kl": float(agent._compute_kl_after_update(context, op)),
        }
        trace = []
        updates = 0
        for _ in range(20):
            batch = next(iter(agent.data_buffer))
            context = agent._prepare_kl_context(batch, op)
            lr = agent.optim.param_groups[0]["lr"]
            kl_before = float(agent._compute_kl_after_update(context, op))
            loss, updates = agent.update(batch, updates)
            kl = agent._compute_kl_after_update(context, op)
            agent._maybe_adjust_lr(kl, agent.config.optim)
            trace.append(
                {
                    "learning_rate_used": lr,
                    "kl_before_update": kl_before,
                    "kl_after_update": float(kl),
                    "learning_rate_next": agent.optim.param_groups[0]["lr"],
                    "normalizers": describe(agent.policy),
                }
            )
        report["actual_update_trace"] = trace
        report["optimizer_updates"] = updates
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0
