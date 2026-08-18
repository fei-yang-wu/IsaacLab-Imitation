"""Method cards: the objective, the interface shapes, and where the code lives.

Equations are MathML so the page stays self-contained -- no script, no font,
no stylesheet fetched from anywhere. Every card names the module that
implements it, because a formula written in a report drifts from the code and
the code is the authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Equation:
    """One display equation with a text fallback and its symbol glossary."""

    mathml: str
    plain: str
    caption: str = ""
    where: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MethodCard:
    """A named approach: what it optimizes, what it publishes, where it lives."""

    id: str
    title: str
    kicker: str
    blurb: str
    equations: tuple[Equation, ...] = ()
    shapes: tuple[tuple[str, str, str], ...] = ()
    source: str = ""
    caveat: str = ""
    tags: tuple[str, ...] = field(default=())


_TRACKER = MethodCard(
    id="tracker_ipmd",
    title="Low-level tracker",
    kicker="50 Hz whole-body policy, frozen at planner time",
    blurb=(
        "The tracker is an IPMD policy over the G1 whole body. It sees the robot "
        "proprioceptive observation and one command from the interface, and it "
        "runs at 50 Hz. A command is held for H control steps, so the interface "
        "publishes at 50/H Hz while the policy still acts every step. When a "
        "planner drives it, the policy state dict is restored strictly, frozen, "
        "and put in evaluation mode; the page records its SHA-256 for every row."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<msub><mi>a</mi><mi>t</mi></msub><mo>&#8764;</mo>"
                "<msub><mi>&#960;</mi><mi>&#952;</mi></msub>"
                "<mo>(</mo><mo>&#8901;</mo><mo>&#8739;</mo>"
                "<msub><mi>o</mi><mi>t</mi></msub><mo>,</mo>"
                "<msub><mi>c</mi><mi>k</mi></msub><mo>)</mo>"
                "<mo>,</mo><mspace width='1em'/>"
                "<mi>k</mi><mo>=</mo>"
                "<mo>&#8970;</mo><mi>t</mi><mo>/</mo><mi>H</mi><mo>&#8971;</mo>"
                "</mrow></math>"
            ),
            plain="a_t ~ pi_theta(. | o_t, c_k),  k = floor(t / H)",
            caption="One command index per hold window; the policy still acts every step.",
            where=(
                ("o_t", "robot proprioceptive observation at 50 Hz"),
                ("c_k", "interface command for hold window k"),
                ("H", "hold length in control steps (10 for the current tracker)"),
            ),
        ),
    ),
    shapes=(
        ("policy parameters", "8,474,170", "frozen under every planner row"),
        ("control rate", "50 Hz", "500 control steps per 10 s episode"),
        ("interface rate", "5 Hz", "at H = 10"),
    ),
    source="RLOpt/rlopt/agent/ipmd/",
    tags=("low-level",),
)


_DIFFSR = MethodCard(
    id="diffsr",
    title="DiffSR latent command",
    kicker="successor-representation objective on the macro window",
    blurb=(
        "DiffSR learns the command space from a successor-measure objective "
        "rather than from reconstruction. The successor measure is factorized "
        "bilinearly into a state-plus-latent embedding and a state embedding, so "
        "a latent z is scored by how well it indexes where the macro window ends "
        "up, not by how exactly it replays the window. The encoder input is the "
        "root_qpos frame: joint positions plus root pose, 380 values wide."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<msup><mi>M</mi><mi>&#960;</mi></msup>"
                "<mo>(</mo><msup><mi>s</mi><mo>&#8242;</mo></msup>"
                "<mo>&#8739;</mo><mi>s</mi><mo>,</mo><mi>z</mi><mo>)</mo>"
                "<mo>&#8776;</mo>"
                "<msup><mi>&#966;</mi><mo>&#8868;</mo></msup>"
                "<mo>(</mo><mi>s</mi><mo>,</mo><mi>z</mi><mo>)</mo>"
                "<mspace width='0.15em'/>"
                "<mi>&#956;</mi><mo>(</mo>"
                "<msup><mi>s</mi><mo>&#8242;</mo></msup><mo>)</mo>"
                "</mrow></math>"
            ),
            plain="M^pi(s' | s, z) ~= phi(s, z)^T mu(s')",
            caption=(
                "Bilinear successor factorization. The loss that fits it is "
                "BilinearSR.compute_loss; this card states the factorization, not "
                "the estimator."
            ),
            where=(
                ("s", "macro-window state, the 380-wide root_qpos frame"),
                ("z", "published latent command"),
                (
                    "phi, mu",
                    "learned embeddings of (state, latent) and successor state",
                ),
            ),
        ),
    ),
    shapes=(
        ("encoder input", "380", "root_qpos frame: qpos + root pose"),
        ("macro window", "10 frames", "current plus nine future"),
    ),
    source="RLOpt/rlopt/agent/hl_skill_diffsr.py",
    caveat=(
        "A v2 checkpoint from before 2026-08-04 was trained on the 670-wide "
        "full-body frame instead. Pairing an old encoder with the current default "
        "fails loudly and must not be forced."
    ),
    tags=("interface", "latent"),
)


_FSQ64 = MethodCard(
    id="fsq64",
    title="FSQ-64 discrete interface",
    kicker="the quantizer output is the command",
    blurb=(
        "The SONIC-style FSQ bottleneck publishes the lattice value itself. There "
        "is no learned projection between the quantizer and the command boundary, "
        "so the tracker observes lattice points directly and the interface is "
        "genuinely quantized rather than merely trained through a quantizer. "
        "Sixty-four dimensions at thirty-two levels each is a 2^320 lattice, so "
        "no flat code index exists and code-usage statistics are per dimension."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<mi>z</mi><mo>=</mo>"
                "<mfrac>"
                "<mrow><mi>round</mi><mo>(</mo>"
                "<mi>bound</mi><mo>(</mo>"
                "<msub><mi>z</mi><mi>e</mi></msub><mo>)</mo><mo>)</mo></mrow>"
                "<mrow><mo>&#8970;</mo><mi>L</mi><mo>/</mo><mn>2</mn><mo>&#8971;</mo></mrow>"
                "</mfrac>"
                "<mo>&#8712;</mo>"
                "<msup>"
                "<mrow><mo>[</mo><mo>&#8722;</mo><mn>1</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow>"
                "<mn>64</mn></msup>"
                "</mrow></math>"
            ),
            plain="z = round(bound(z_e)) / floor(L/2)  in  [-1, 1]^64",
            caption=(
                "Straight-through in the backward pass: dividing by a constant "
                "preserves the estimator and scales the gradient by 1/floor(L/2)."
            ),
            where=(
                ("z_e", "continuous encoder output before quantization"),
                ("L", "levels per dimension, 32 for the current arm"),
                ("bound", "the FSQQuantizer bounding map onto the lattice range"),
            ),
        ),
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<mo>&#8739;</mo><mi>&#119985;</mi><mo>&#8739;</mo>"
                "<mo>=</mo>"
                "<msup><mn>32</mn><mn>64</mn></msup>"
                "<mo>=</mo>"
                "<msup><mn>2</mn><mn>320</mn></msup>"
                "</mrow></math>"
            ),
            plain="|codebook| = 32^64 = 2^320",
            caption="Too large for a flat index, so perplexity is not reported.",
        ),
    ),
    shapes=(
        ("published command", "64", "lattice values, identity projection"),
        ("levels per dimension", "32", "SONIC's 64 x 32 lattice"),
        (
            "hold",
            "10 control steps",
            "with a 2-wide sin_cos phase channel, code_period 10",
        ),
    ),
    source="RLOpt/rlopt/agent/hl_skill_encoder.py :: SONICFSQSkillEncoder",
    tags=("interface", "latent", "discrete"),
)


_Z256 = MethodCard(
    id="z256",
    title="Continuous 256-D interface",
    kicker="deterministic latent, no bottleneck at the boundary",
    blurb=(
        "The continuous control arm for the discrete-versus-continuous question. "
        "The encoder publishes a 256-value real vector with no quantization, so "
        "every point in R^256 is an admissible command. It is welded to its own "
        "tracker, trained under the same protocol as the FSQ tracker, which is "
        "why a small part of any measured gap is tracker and not interface."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<mi>z</mi><mo>=</mo>"
                "<msub><mi>f</mi><mi>&#966;</mi></msub>"
                "<mo>(</mo><mi>s</mi><mo>)</mo>"
                "<mo>&#8712;</mo>"
                "<msup><mi>&#8477;</mi><mn>256</mn></msup>"
                "</mrow></math>"
            ),
            plain="z = f_phi(s)  in  R^256",
            caption="Identity bottleneck: the encoder output is the command.",
        ),
    ),
    shapes=(("published command", "256", "unconstrained real vector"),),
    source="RLOpt/rlopt/agent/hl_skill_encoder.py :: DeterministicSkillEncoder",
    tags=("interface", "latent", "continuous"),
)


_EXPLICIT = MethodCard(
    id="explicit_packet",
    title="Explicit command packet",
    kicker="the baseline every humanoid VLA publishes",
    blurb=(
        "The comparison arm: instead of a latent, the planner publishes ten "
        "vanilla tracker commands, current plus nine future frames, term-major. "
        "Anchors are re-expressed against the current robot anchor, and the ten "
        "slots are consumed once each before renewal. The direct 50 Hz vanilla "
        "tracker receiving a fresh expert command is the low-level ceiling for "
        "this arm, not a planner row."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<mn>10</mn><mo>&#215;</mo><mn>58</mn>"
                "<mo>+</mo><mn>10</mn><mo>&#215;</mo><mn>3</mn>"
                "<mo>+</mo><mn>10</mn><mo>&#215;</mo><mn>6</mn>"
                "<mo>=</mo><mn>580</mn><mo>+</mo><mn>30</mn><mo>+</mo><mn>60</mn>"
                "<mo>=</mo><mn>670</mn>"
                "</mrow></math>"
            ),
            plain="10x58 + 10x3 + 10x6 = 580 + 30 + 60 = 670",
            caption="Term-major packet width against 64 for FSQ and 256 for z256.",
            where=(
                ("expert_motion", "580 values, ten 58-wide motion frames"),
                ("anchor_pos", "30 values, ten 3-vectors"),
                ("anchor_ori", "60 values, ten 6-D rotations"),
            ),
        ),
    ),
    shapes=(("published command", "670", "term-major [580, 30, 60]"),),
    source="wiki/causal-interface-paper-plan.md",
    tags=("interface", "explicit"),
)


_PLANNER = MethodCard(
    id="flow_planner",
    title="Flow-matching planner head",
    kicker="verbatim GR00T N1.7 action head",
    blurb=(
        "The planner is the unmodified GR00T N1.7 action head with a warm-started "
        "trunk. It sees only causal information: nine past robot frames plus the "
        "current one at 93 values each, and a language embedding of the goal. It "
        "never sees future reference data. The head is trained by flow matching "
        "to transport noise onto the latent target, and sampled by Euler "
        "integration of the learned velocity field."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<mi>&#8466;</mi><mo>=</mo>"
                "<msub><mi>&#120124;</mi>"
                "<mrow><mi>&#964;</mi><mo>,</mo>"
                "<msub><mi>x</mi><mn>0</mn></msub><mo>,</mo>"
                "<msub><mi>x</mi><mn>1</mn></msub></mrow></msub>"
                "<msup><mrow><mo>&#8741;</mo>"
                "<msub><mi>v</mi><mi>&#968;</mi></msub>"
                "<mo>(</mo><msub><mi>x</mi><mi>&#964;</mi></msub><mo>,</mo>"
                "<mi>&#964;</mi><mo>,</mo><mi>c</mi><mo>)</mo>"
                "<mo>&#8722;</mo>"
                "<mo>(</mo><msub><mi>x</mi><mn>1</mn></msub>"
                "<mo>&#8722;</mo><msub><mi>x</mi><mn>0</mn></msub><mo>)</mo>"
                "<mo>&#8741;</mo></mrow><mn>2</mn></msup>"
                "</mrow></math>"
            ),
            plain="L = E_{tau,x0,x1} || v_psi(x_tau, tau, c) - (x_1 - x_0) ||^2",
            caption=(
                "Conditional flow matching on the straight path "
                "x_tau = (1 - tau) x_0 + tau x_1."
            ),
            where=(
                ("x_0", "Gaussian noise"),
                (
                    "x_1",
                    "latent target: the oracle command the tracker would have received",
                ),
                ("c", "causal robot history plus the language goal embedding"),
                ("tau", "flow time in [0, 1], 16 Euler steps at inference"),
            ),
        ),
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<msub><mi>x</mi><mrow><mi>obs</mi></mrow></msub>"
                "<mo>&#8712;</mo>"
                "<msup><mi>&#8477;</mi>"
                "<mrow><mn>10</mn><mo>&#215;</mo><mn>93</mn></mrow></msup>"
                "<mo>,</mo><mspace width='1em'/>"
                "<msub><mi>x</mi><mn>1</mn></msub>"
                "<mo>&#8712;</mo>"
                "<msup><mi>&#8477;</mi>"
                "<mrow><mn>3</mn><mo>&#215;</mo><mn>64</mn></mrow></msup>"
                "</mrow></math>"
            ),
            plain="x_obs in R^{10x93},  x_1 in R^{3x64}",
            caption=(
                "Action horizon 3: the head predicts three latents, each held ten "
                "control steps."
            ),
        ),
    ),
    shapes=(
        ("planner input", "10 x 93", "nine past frames plus current, causal only"),
        ("action horizon", "3", "latents per head call"),
        ("language", "384", "all-MiniLM-L6-v2 sentence embedding"),
        ("head parameters", "2,302,016", "reported by the evaluation metadata"),
    ),
    source="external/Isaac-GR00T (pinned) + RLOpt adapters",
    caveat=(
        "A loss mask-normalization defect and an undisclosed "
        "attend_text_every_n_blocks=1 masking change were found in the parity "
        "check; both are unresolved against upstream."
    ),
    tags=("planner",),
)


_ENSEMBLING = MethodCard(
    id="temporal_ensembling",
    title="Exponential temporal ensembling",
    kicker="the only inference knob that improved both axes",
    blurb=(
        "Head calls overlap: at control step t several earlier calls have already "
        "predicted a latent for t. Averaging them with exponentially decaying "
        "weights improved MPJPE and survival together. Sample averaging and extra "
        "ODE steps did not, because they redraw at one state while ensembling "
        "blends estimates made from different states."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<msub><mover><mi>z</mi><mo>&#770;</mo></mover><mi>t</mi></msub>"
                "<mo>=</mo>"
                "<mfrac>"
                "<mrow><munder><mo>&#8721;</mo><mi>j</mi></munder>"
                "<msup><mi>&#961;</mi><mi>j</mi></msup>"
                "<msubsup><mi>z</mi><mi>t</mi>"
                "<mrow><mo>(</mo><mi>t</mi><mo>&#8722;</mo><mi>j</mi><mo>)</mo></mrow>"
                "</msubsup></mrow>"
                "<mrow><munder><mo>&#8721;</mo><mi>j</mi></munder>"
                "<msup><mi>&#961;</mi><mi>j</mi></msup></mrow>"
                "</mfrac>"
                "<mo>,</mo><mspace width='1em'/>"
                "<mi>&#961;</mi><mo>=</mo><mn>0.5</mn>"
                "</mrow></math>"
            ),
            plain="z_hat_t = sum_j rho^j z_t^{(t-j)} / sum_j rho^j,  rho = 0.5",
            caption="Decay 0.5 over the head's overlapping predictions for step t.",
            where=(
                ("z_t^{(t-j)}", "prediction for step t made by the call at step t-j"),
                ("rho", "decay; 0.5 was the published setting, 0.25 was also measured"),
            ),
        ),
    ),
    source="imitation_experiments.planner",
    tags=("planner", "inference"),
)


_MPJPE = MethodCard(
    id="mpjpe",
    title="How the headline number is reduced",
    kicker="episode-mean, root-relative MPJPE",
    blurb=(
        "The headline MPJPE is the unweighted mean over episodes of each "
        "episode's root-relative mean per-joint position error. With a balanced "
        "run it equals the motion-averaged value. Two other reductions sit in the "
        "same artifact and read differently: transition-weighted, which lets long "
        "episodes dominate, and successful-only, which drops failed episodes and "
        "so flatters an arm where it is weakest. The page shows all three."
    ),
    equations=(
        Equation(
            mathml=(
                '<math display="block"><mrow>'
                "<mi>MPJPE</mi><mo>=</mo>"
                "<mfrac><mn>1</mn><mi>E</mi></mfrac>"
                "<munderover><mo>&#8721;</mo>"
                "<mrow><mi>e</mi><mo>=</mo><mn>1</mn></mrow><mi>E</mi></munderover>"
                "<mfrac><mn>1</mn>"
                "<mrow><msub><mi>T</mi><mi>e</mi></msub><mi>J</mi></mrow></mfrac>"
                "<munder><mo>&#8721;</mo><mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></munder>"
                "<mrow><mo>&#8741;</mo>"
                "<msub><mover><mi>p</mi><mo>&#770;</mo></mover><mrow><mi>j</mi><mi>t</mi></mrow></msub>"
                "<mo>&#8722;</mo>"
                "<msub><mi>p</mi><mrow><mi>j</mi><mi>t</mi></mrow></msub>"
                "<mo>&#8741;</mo></mrow>"
                "</mrow></math>"
            ),
            plain=(
                "MPJPE = (1/E) sum_e (1/(T_e J)) sum_{t,j} || p_hat_{jt} - p_{jt} ||"
            ),
            caption="Positions are root-relative, so root drift does not enter.",
            where=(
                ("E", "episodes in the run"),
                ("T_e", "valid transitions in episode e"),
                ("J", "tracked bodies"),
            ),
        ),
    ),
    shapes=(
        (
            "survival",
            "fall-only",
            "SONIC termination profile, no push, randomization on",
        ),
        ("noise band", "~15% relative", "smaller differences are directional only"),
    ),
    source=("imitation_experiments/reporting/records.py :: _reduce_per_motion"),
    tags=("metric",),
)


CARDS: dict[str, MethodCard] = {
    card.id: card
    for card in (
        _TRACKER,
        _DIFFSR,
        _FSQ64,
        _Z256,
        _EXPLICIT,
        _PLANNER,
        _ENSEMBLING,
        _MPJPE,
    )
}


def cards_for(ids: tuple[str, ...]) -> list[MethodCard]:
    """Return the requested cards in order, failing on an unknown id."""
    unknown = [card_id for card_id in ids if card_id not in CARDS]
    if unknown:
        raise KeyError(f"Unknown method card(s) {unknown}; available: {sorted(CARDS)}")
    return [CARDS[card_id] for card_id in ids]
