# Whole-Body VLA and Latent-Action Literature Review

Last reviewed: 2026-07-16.

This page records the literature behind the paper's interface comparison. Its
purpose is to keep three things separate:

1. what a published method actually predicts and executes;
2. what that method teaches us about our experiment design; and
3. whether we are reproducing the method or only testing the same design idea.

The current project does **not** reproduce every named system. The main paper
experiment is a controlled comparison between our DiffSR latent interface and
the exact explicit command sequence consumed by our vanilla whole-body
tracker. A method name should appear on a result row only when we use its
released model or faithfully reproduce its native inputs, output
representation, timing, controller, and training procedure.

## The Important Terminology Distinction

The phrase *latent action* is used for several different ideas in the
literature:

- **Deployed latent control interface:** the high-level policy outputs a
  learned code that a separate low-level controller consumes. SONIC's
  universal motion tokens, LeVERB's latent verbs, and our DiffSR code belong
  here.
- **Latent labels for pretraining:** a model learns action-like tokens from
  video, but later fine-tuning maps the model to explicit robot actions. LAPA
  is the clearest example. GR00T N1 also uses latent action labels to learn
  from actionless video.
- **Latent trajectory model:** a planner compresses state-action trajectories
  internally, but the deployed robot interface need not be latent. H-GAP is
  in this family.
- **Policy or task embedding in a behavior foundation model:** a latent `z`
  indexes a family of policies and is chosen once per task or episode, as in
  forward-backward models such as Meta Motivo. This is adjacent to our
  setting, but the code is a task selector, not a 5 Hz closed-loop command
  stream produced by a causal planner.

Our research question concerns the first case: whether a learned code is a
better **deployed high-to-low-level interface** than an explicit future
full-body command packet.

## Core Method Comparison

| Method | High-level input | Deployed high-level output | High/low rate | What it tests | Relationship to our work |
| --- | --- | --- | --- | --- | --- |
| **Our main comparison** | Causal `10 x 93` robot history; optional explicit language goal | Either a continuous 256-value DiffSR code or the exact 670-value ten-frame vanilla command packet | 5 Hz / 50 Hz | Whether the interface alone changes planner learning difficulty when the rest of the protocol is matched | Main controlled experiment |
| **SONIC VLA integration** | Vision, language, and robot context through GR00T N1.5 | 78 values: a 64-value discrete universal motion token plus 14 hand joints; explicit ablation uses 81-value SMPL pose plus hands | VLA/low-level timing is system-dependent; tracker runs at 50 Hz | Whether learned universal motion tokens are easier for a VLA to predict than explicit poses | Closest published quantitative evidence for our hypothesis, but not a native reproduction |
| **SONIC kinematic planner** | Four generated context frames plus operator motion commands | Full-body trajectory, up to 64 frames at 30 fps, resampled for the 50 Hz tracker | Up to 10 Hz / 50 Hz | Real-time generation of explicit whole-body reference trajectories | Motivates the strong explicit full-body baseline; distinct from SONIC's token VLA experiment |
| **HuMI** | Camera streams and proprioception | Receding-horizon task-space keypoints and gripper commands; default keypoints are both grippers and pelvis, with feet optional | 5 Hz / 50 Hz | Whether compact task-space trajectories can drive whole-body manipulation | Best reference for an EE/pelvis chunk diagnostic, not the main full-body baseline |
| **WholeBodyVLA** | Egocentric image and language | Two learned discrete latent action streams, decoded into dual-arm joint actions and locomotion commands | 10 Hz / 50 Hz locomotion controller | Unified visual latent actions for locomotion and manipulation | Same broad latent-interface motivation; its visual latent and split decoder differ from DiffSR |
| **LeVERB** | Egocentric/third-person vision and language | A continuous 256-value latent verb consumed by a distilled whole-body controller | Hierarchical slow/fast system; exact numeric rates are missing from the paper's HTML conversion | A vision-language policy and whole-body controller connected through a learned latent vocabulary | Closest conceptual predecessor to our language-conditioned Phase 5 study |
| **SENTINEL** | Language and multi-scale proprioceptive history | Flow-matched chunks of low-level robot actions, with an optional residual action head | Action interface at 50 Hz; asynchronous chunk generation on hardware | Fully end-to-end language-to-action whole-body control without an intermediate motion command | Strong direct-action alternative, but it intentionally removes the frozen low-level interface we are studying |
| **LangWBC** | Language and proprioceptive history | Low-level actions from one distilled CVAE policy | One end-to-end control policy | Language-conditioned whole-body control with no separate high-level command | Relevant alternative system design, not an interface row |
| **GR00T N1** | Images, language, and embodiment state | A flow-matching DiT directly predicts embodiment-specific action chunks | System 2 at 10 Hz; action module targets high-frequency control | A general VLA trained across heterogeneous embodiments and data sources | Motivates our flow-matching planner family, but original N1 is not a single G1 whole-body command baseline |
| **LAPA** | Image and task description | Latent action tokens during pretraining; explicit end-effector deltas after robot-action fine-tuning | Application-dependent | Learning action-like supervision from actionless video | Relevant latent-action lineage, but its latent is not the deployed robot interface |

Rates should not be treated as interchangeable across papers. Some papers
report a model inference rate, others an action sampling rate, and others the
low-level control loop. A fair native comparison must record all three.

## Detailed Notes

### SONIC

[SONIC](https://arxiv.org/abs/2511.07820) trains a universal Unitree G1 motion
tracker on more than 100 million motion frames. Its control policy runs at
50 Hz. The paper studies two high-level uses of the tracker that are relevant
to us but should not be conflated.

First, its real-time kinematic planner generates explicit full-body motion.
It replans at up to 10 Hz, predicts as many as 64 frames at 30 fps from four
recent generated frames and operator commands, and resamples the result to
50 Hz for tracking. This is evidence that an explicit full-body trajectory is
a serious high-level interface, not merely a weak ablation.

Second, the GR00T N1.5 VLA integration predicts SONIC's discrete universal
motion tokens. For whole-body tasks, the action has 64 token values and 14
hand-joint values. The paper compares this with an 81-value explicit SMPL-pose
and hand action. Across three reported tasks, token prediction averages 68%
success and explicit SMPL prediction averages 27%. The gap is especially
large on the longest multi-step task.

This supports our main hypothesis, but it does not settle it. SONIC's
comparison changes the output representation and uses SONIC's quantizer,
tracker, VLA training, and task data. Our experiment asks whether the advantage
remains when the high-level observations, planner family, positive sample
count, optimizer budget, publication schedule, and evaluation starts are held
fixed.

SONIC also supports our metric choices: it reports local MPJPE, tracking
success, and velocity and acceleration distances. Its released controller is
available in [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).

### HuMI

[HuMI](https://arxiv.org/abs/2602.06643) collects robot-free whole-body
manipulation demonstrations and trains a visual high-level policy. The policy
runs at 5 Hz and produces a receding-horizon trajectory of task-space
keypoints and gripper commands. A whole-body controller tracks those targets
at 50 Hz.

The default interface uses the two grippers and pelvis. Feet can also be
included. The high-level model is a flow-matching Diffusion Policy trained with
a 48-step action horizon at a 20 Hz data frequency and ten denoising steps.
These are training/action-sequence settings; they should not be mislabeled as
the 5 Hz publication rate.

HuMI's interface ablation is directly useful to our baseline choice:

- end-effector-only targets do not reliably distinguish reaching from moving
  the body toward an object;
- adding pelvis information substantially improves whole-body intent;
- adding feet can help but is not always needed beyond grippers plus pelvis;
- global targets for non-visually grounded keypoints accumulate drift, so the
  paper uses relative tracking and continuity rules across chunks.

This is why EE-only should not replace the explicit full-body row in our main
comparison. It would remove information and make a latent win difficult to
interpret. A HuMI-style `EE + pelvis` packet is still a valuable appendix
diagnostic: it tests whether a carefully chosen lower-dimensional explicit
interface can close the gap without learning a latent code.

HuMI also handles chunk continuity differently from our frozen protocol. It
forms the next relative target from the previous target rather than the
lagging achieved end-effector pose. Our explicit packet is expressed against
the current achieved robot anchor. This difference should be reported if we
run a HuMI-style diagnostic; it should not be silently folded into the main
baseline.

### WholeBodyVLA

[WholeBodyVLA](https://arxiv.org/abs/2512.11047) learns separate manipulation
and locomotion latent action models from consecutive video frames. A VLA
predicts both discrete latent streams from an egocentric image and language.
A lightweight execution decoder turns them into upper-body joint actions and
locomotion commands; the VLA/decoder operates at 10 Hz and the locomotion RL
policy at 50 Hz.

The shared idea is that a learned compact representation may be easier for a
high-level model to predict than raw robot control. The main difference is the
source and role of the representation. WholeBodyVLA learns visual dynamics
tokens and splits locomotion from manipulation. Our DiffSR code is learned
from robot motion transitions and directly conditions one frozen whole-body
controller. Native comparison would require visual observations, its two
tokenizers, decoder, locomotion controller, and task suite. The released code
is [OpenDriveLab/WholebodyVLA](https://github.com/OpenDriveLab/WholebodyVLA).

### LeVERB

[LeVERB](https://arxiv.org/abs/2506.13751) is the closest conceptual prior to
our language-conditioned study. A high-level vision-language model produces a
256-value latent verb, and a separately trained whole-body controller turns
that code and proprioception into joint commands. The high-level System 2 has
about 102.6 million trainable parameters. The matching 256-value interface is
not evidence that its latent has the same meaning as DiffSR; it does make
planner size and interface bandwidth an especially useful comparison.

The latent vocabulary is learned with a residual CVAE. During training, a
privileged trajectory encoder supplies future-motion detail for
reconstruction, while the deployed vision-language prior sees only the
current images and language. The latent space is frozen before the low-level
student is distilled from privileged tracking teachers with DAgger. The paper
evaluates more than 150 vision-language whole-body tasks and reports 58.5%
overall success, including real Unitree G1 demonstrations.

The methodological lesson for us is not to prohibit future motion during
supervised representation learning. Future motion may define a target code.
The causal rule is that future reference information must not enter the
deployed high-level planner. Our encoder targets and oracle labels may use the
reference; the Phase 5 planner input may only use achieved robot history and
the explicit language goal.

LeVERB differs from our work in several important ways. It learns the latent
jointly around a vision-language reconstruction objective, samples a latent
distribution during low-level training, and evaluates semantic scene tasks.
Our current Phase 5 goals name fixed motions and deliberately remove vision so
we can isolate whether the control interface changes planner learning.

### GR00T N1 and Direct Action-Chunk VLAs

[GR00T N1](https://arxiv.org/abs/2503.14734) combines a vision-language
backbone with a flow-matching Diffusion Transformer action module. The released
2.2-billion-parameter model samples a chunk of 16 high-frequency actions. Its
action representation depends on the embodiment: examples include
end-effector deltas and combinations of arm, hand, waist, and neck joint
targets.

This makes GR00T a useful planner-design reference, but not a single native
whole-body baseline that can be represented by our 670-value packet. Our
flow-matching Transformer tests the same broad choice—continuous chunk
generation with flow matching—inside a much smaller controlled state-only
experiment. Later GR00T integrations with SONIC should be cited through the
SONIC experiment rather than retroactively attributed to the original N1
paper.

Two other direct chunk methods define the bounded planner architecture study:

- [Diffusion Policy](https://arxiv.org/abs/2303.04137) motivates a conditional
  diffusion model over receding-horizon action sequences.
- [ACT](https://arxiv.org/abs/2304.13705) motivates a deterministic/generative
  Transformer that predicts action chunks.
- [pi0](https://arxiv.org/abs/2410.24164) is another important flow-matching
  VLA reference, but it is a general robot policy rather than a native
  humanoid whole-body controller.

These references justify comparing flow matching, diffusion, and
deterministic chunk prediction. The comparison should use the same causal
input, output target, actual parameter count, data, training updates, and
evaluation starts for both interfaces. It is an architecture-family study,
not a claim that our small state-only models reproduce GR00T, ACT, or pi0.

### End-to-End Humanoid Language-Action Policies

[SENTINEL](https://arxiv.org/abs/2511.19236) removes the intermediate
high-level motion representation entirely. A Transformer consumes language
and multi-scale proprioceptive history, and a flow-matching action expert
predicts chunks of low-level robot actions. Its recent history contains ten
steps at 50 Hz; a coarser history covers ten seconds at 4 Hz. Only part of each
predicted chunk is executed before replanning. For hardware, an optional
residual head trained with PPO corrects the chunk while preserving its intended
motion.

SENTINEL is especially relevant to our planner scaling study. It reports 60M,
200M, and 600M model variants, with much stronger language alignment at larger
sizes and a large success drop for its 60M model. This supports reporting the
smallest model that reaches a fixed target rather than evaluating only one
planner size. Its direct-action design is not one of our two main rows because
it has no shared frozen low-level controller; adding it would change both the
interface and the control architecture.

[LangWBC](https://arxiv.org/abs/2504.21738) is another end-to-end alternative.
It distills a motion-tracking teacher into a CVAE student that maps language
and proprioceptive history directly to whole-body control actions. The latent
space is internal to the single policy rather than a published command between
separately evaluated high- and low-level systems. It therefore demonstrates
language-conditioned motion generation, but it does not answer whether a
latent is a better interface to a fixed controller.

[Humanoid-VLA](https://arxiv.org/abs/2502.14795) represents the hierarchical
motion-generation direction: language and egocentric visual context are used
to generate whole-body motion that a controller executes. It is relevant when
we later add vision and scene-level tasks. For the current no-vision study, the
stronger and more controlled explicit reference remains the exact command
packet consumed by our qualified vanilla tracker.

### LAPA and Video-Derived Latent Actions

[LAPA](https://arxiv.org/abs/2410.11758) first learns discrete latent actions
between current and future video frames with a VQ-VAE. A vision-language model
then predicts those tokens from an image and task description. Finally, small
action-labeled robot datasets map the pretrained model to explicit
end-effector delta actions.

LAPA establishes that latent action labels can make actionless human video
useful for VLA pretraining and may form a shared representation across
embodiments. It does not test whether a latent is a better deployed interface
to a frozen low-level robot controller. The paper also observes that visual
latent actions can encode camera motion. Our state-transition representation
avoids that specific ambiguity, although it cannot directly exploit internet
video at LAPA's scale.

### Earlier Latent Skill and Trajectory Models

The latent-interface idea also has a control lineage that predates recent VLA
work:

- [Adversarial Skill Embeddings (ASE)](https://arxiv.org/abs/2205.01906)
  learns reusable continuous motor-skill embeddings from unstructured motion
  data and trains downstream task policies to command them.
- [CALM](https://arxiv.org/abs/2305.02195) learns a semantically structured
  latent motion representation together with a control policy, then uses that
  representation for higher-level tasks.
- [H-GAP](https://arxiv.org/abs/2312.02682) learns a generative latent model of
  humanoid state-action trajectories and uses it for planning.
- [HOVER](https://arxiv.org/abs/2410.21229) is relevant low-level context: it
  studies a generalist humanoid controller that accepts several explicit
  command modes, but it is not a VLA planner-interface comparison.

These works support the premise that motion representations can expose a
useful control space. Recent VLA papers add the question of whether a visual or
language high-level model can learn that interface efficiently.

### Explicit Trajectory Generation as a Deployed Interface

[BeyondMimic](https://arxiv.org/abs/2508.08241) is the strongest recent
evidence that explicit whole-body motion is a serious deployed high-level
interface: sim-to-real trackers for dynamic skills, then a unified
guided-diffusion model steered by test-time cost functions for waypoint
navigation, joystick teleoperation, and obstacle avoidance on real hardware.
Our command-space notes already describe the full-body window as
"BeyondMimic-style", so cite it as motivation for the strong explicit row.
Verify its exact deployed output space (state, action, or joint chunks)
before any quantitative comparison.
[Diffuse-CLoC](https://arxiv.org/abs/2503.11801) similarly plans with guided
diffusion over look-ahead states for physics characters. Neither is
reproduced here.

### Physics-Based Character Skill Latents

This lineage predates and directly anticipates our latent row. Each entry
deploys a learned latent to a physics controller at runtime, so the
"pretraining only" caveat that applies to LAPA does not apply here.

- [NPMP](https://arxiv.org/abs/1811.11711) introduced the latent motor-module
  bottleneck: thousands of expert policies compressed offline into one latent
  space that downstream controllers command.
- [ControlVAE](https://arxiv.org/abs/2210.06063) learns a
  world-model-supervised VAE skill space that high-level policies reuse for
  downstream tasks.
- [PHC](https://arxiv.org/abs/2305.06456) scales one tracker to ten thousand
  clips and is the tracker behind
  [PULSE](https://arxiv.org/abs/2310.04582), which distills it into a
  32-value probabilistic latent through a variational information bottleneck
  (ICLR 2024 spotlight). PULSE is the reconstruction-family method closest to
  a latent-objective ablation for us, and its 32-value width is a useful
  bandwidth contrast to our 256.
- [MaskedMimic](https://arxiv.org/abs/2409.14393) reframes the interface as
  masked motion inpainting: flexibility comes from partial explicit goals
  rather than a learned code. It is the strongest "neither row" design and
  deserves one sentence in related work.
- [PADL](https://arxiv.org/abs/2301.13868) and CALM show that latent skill
  spaces support semantically meaningful interpolation between skills.
- [FLD](https://arxiv.org/abs/2402.13820) imposes explicit periodic Fourier
  structure on a motion latent; the most direct precedent for adding
  structure to a skill latent.
- [Versatile hybrid latents](https://arxiv.org/abs/2503.12814) split a
  discrete part-wise latent from a continuous whole-body latent and compose
  body-part skills at runtime.
- [InsActor](https://arxiv.org/abs/2312.17135) stacks a language-conditioned
  diffusion planner on a CVAE skill space — the graphics analog of our
  Phase 5 stack.

## The Representation-Learning Lineage

The sections above cover the robotics and VLA axis. An ICLR audience will
also judge the DiffSR skill encoder against the representation-learning
literature, which this project must cite and position. The standing check is
the same: is the latent a deployed closed-loop command (our question), a
task or policy index, or a pretraining scaffold?

### Spectral and low-rank factorizations of dynamics

Our skill objective is an instance of this line: a transition kernel is
factorized through learned features so planning-relevant quantities become
(bi)linear in the representation.

- [SPEDER](https://arxiv.org/abs/2208.09515) learns a spectral decomposition
  of the state-action transition kernel with sample-efficiency guarantees.
- [LV-Rep](https://arxiv.org/abs/2212.08765) gives the latent-variable view
  of the same factorization with tractable variational learning.
- [Diff-SR](https://arxiv.org/abs/2406.16121) is the direct ancestor of our
  objective: spectral representations extracted from a diffusion model of the
  transition kernel, avoiding expensive sampling at decision time. Our delta
  is to condition the factorized score on a skill code produced by a
  future-window encoder and to deploy that code as the command of a frozen
  whole-body tracker.
- [Spectral Representation-based RL](https://arxiv.org/abs/2512.15036) is a
  recent unification of this family and a convenient citation shortcut.

### Forward-backward representations and behavior foundation models

[FB representations](https://arxiv.org/abs/2103.07945) and
[Does Zero-Shot RL Exist?](https://arxiv.org/abs/2209.14935) learn a low-rank
successor-measure factorization `F(s,a)ᵀB(s')` together with a z-conditioned
policy family — the nearest representation-learning relative of a deployed
latent control interface.
[Meta Motivo / FB-CPR](https://arxiv.org/abs/2504.11054) applies FB with
imitation grounding to whole-body humanoid control in simulation: tracking,
goal reaching, and reward optimization are all solved zero-shot by prompting
with a latent `z`. This is the closest cousin of our latent row and must be
cited; the precise differences are that its `z` is chosen per task or episode
by encoding a prompt rather than streamed at 5 Hz from causal robot history,
and it makes no controlled latent-versus-explicit interface comparison
against a frozen tracker. [BFM-Zero](https://arxiv.org/abs/2511.04131)
pushes a promptable FB-style behavioral foundation model toward real humanoid
robot control with unsupervised RL, and a
[position paper](https://arxiv.org/abs/2506.20487) argues BFMs are the
next-generation humanoid whole-body control system. Active theory work
continues in [Soft FB](https://arxiv.org/abs/2602.06769) and a
[critical re-examination](https://arxiv.org/abs/2602.11399) of FB claims.

### Skill discovery with predictable or metric structure

- [DADS](https://arxiv.org/abs/1907.01657) discovers skills whose *outcomes
  are easy to predict*. Our encoder is the supervised, inference-side analog:
  it encodes expert futures so the macro endpoint is predictable through a
  factorized score. This one-sentence connection belongs in the paper.
- [METRA](https://arxiv.org/abs/2310.08887) learns a temporal-metric-aware
  latent where directions correspond to diverse behaviors; a candidate
  structure prior for the composition study.
- [HILP](https://arxiv.org/abs/2402.15567) learns a Hilbert representation
  where latent distances are temporal distances and latents act as directions
  for zero-shot goal reaching and hierarchy.

### Hierarchical interface theory

[Near-Optimal Representation Learning for Hierarchical RL](https://arxiv.org/abs/1810.01257)
bounds the sub-optimality of a hierarchical policy as a function of the
goal/command representation — the only formal framework for exactly our
question. Our study is its empirical, physics-grounded instantiation, and a
bound-styled decomposition (planner target error times low-level command
sensitivity) is the candidate skeleton for a mechanism analysis.

### Decoder-free predictive world models (JEPA family)

[I-JEPA](https://arxiv.org/abs/2301.08243) predicts representations rather
than reconstructing observations;
[TD-JEPA](https://arxiv.org/abs/2510.00739) extends latent-predictive
learning to multi-step, policy-conditioned TD targets and formally connects
JEPA-style learning to successor features and FB;
[DINO-WM](https://arxiv.org/abs/2411.04983) plans in a frozen pretrained
feature space. Position DiffSR honestly against this family: training is
decoder-free (no reconstruction objective; encoder-only deployment), but the
denoising target lives in observation space, so it is a factorized diffusion
world model rather than a latent-target JEPA. Claim "JEPA-spirited", not
"JEPA-style".

### Latent action tokenization for imitation

[LAPO](https://arxiv.org/abs/2312.10812) recovers latent actions from
action-free video (ICLR 2024 spotlight);
[VQ-BeT](https://arxiv.org/abs/2403.03181) and
[QueST](https://arxiv.org/abs/2407.15840) learn discrete latent action-chunk
vocabularies that behavior-cloned policies deploy. These are deployed latent
interfaces, but the tokenizer and policy head are co-trained and there is no
frozen RL tracker, so they answer a different question than our comparison.
They matter for the discrete-latent appendix diagnostic alongside SONIC's FSQ
result.

### Structured latent spaces and skill composition (exploratory)

Evidence that skill latents support composition: ASE, CALM, and PADL
interpolation; part-wise hybrid latents; FLD's periodic parameterization; FB
task arithmetic in `z` space; METRA and HILP directional semantics. Our own
composition direction is exploratory. Any added structure (orthogonality,
directional constraints, periodicity) changes the encoder and must be ablated
against the frozen paper encoder, never silently swapped in. If it matures it
becomes one secondary section with closed-loop compositional evaluation
against defined target motions, not a headline claim.

### Found 2026-07-16, not yet read

Triage before citing; record the deployed-interface status for each:
[MotionVLA](https://arxiv.org/abs/2606.15142),
[CLAP](https://arxiv.org/abs/2601.04061),
[ZEST](https://arxiv.org/abs/2602.00401),
[Perceptive Behavior Foundation Model](https://arxiv.org/abs/2606.08059),
[AnyBody](https://arxiv.org/abs/2606.29209),
[WristMimic](https://arxiv.org/abs/2607.06438),
[DreamControl](https://arxiv.org/abs/2509.14353).

## What We Should Compare in This Project

### Main controlled paper comparison

Keep exactly the two current rows:

1. DiffSR's continuous 256-value latent command at 5 Hz.
2. The 670-value packet containing the exact ten vanilla commands that the
   same 50 Hz tracker would otherwise receive directly.

Use the same causal achieved-state history, language token when present,
planner family, sample count, optimization budget, seeds, command renewal
rules, and evaluation starts. Report absolute metrics and performance
normalized by each interface's qualified low-level oracle.

This is stronger causal evidence than comparing our latent with a weak or
under-specified EE command. The vanilla packet contains the same full command
information available to the vanilla expert over the next macro step. If the
explicit planner fails while its oracle succeeds, the difficulty lies in
predicting that command sequence rather than in an intentionally limited
low-level interface.

### Bounded planner comparison

Compare only three planner families already implemented:

1. flow-matching Transformer;
2. diffusion chunk Transformer;
3. deterministic chunk predictor.

For each family, answer both questions:

- At matched actual parameter counts, which interface performs better?
- What is the smallest tested planner that reaches a fixed survival and
  oracle-normalized MPJPE target?

Also report planner-only inference latency and output bandwidth. Keep
demonstration-only and rollout-fine-tuned results separate.

### Focused appendix diagnostics

Only add these after the main pipeline is healthy:

- a HuMI-style `EE + pelvis` trajectory packet, optionally with feet, to test
  whether a compact explicit task-space design closes the gap;
- a discrete/quantized latent ablation if we implement a genuine tokenizer,
  motivated by SONIC's FSQ results;
- a language-goal generalization split once Phase 5 can accurately command
  the named motions without reference leakage.

Candidate additions motivated by the representation-learning lineage. These
extend the current frozen scope and need explicit user approval before
entering any run grid:

- a representation-objective ablation at a fixed 256-value width — one
  reconstruction/CVAE latent (PULSE/LeVERB lineage) and one FSQ/VQ discrete
  latent — each with its own oracle-gated low-level controller, so the
  spectral objective itself, not merely "some latent", carries the claim;
- a matched-bandwidth explicit control: PCA or a linear autoencoder
  compressing the 670-value packet to 256 values, decompressed and consumed
  by the same frozen vanilla tracker, separating output dimensionality from
  learned structure;
- a structured-latent composition probe (interpolation and sequencing between
  named skills) executed by the frozen tracker, reported qualitatively unless
  a closed-loop compositional metric is defined first.

Do not combine every command representation, planner family, model size, data
budget, and language split into one grid. Each diagnostic should answer one
specific question.

### Native SOTA evaluation, later

A true comparison to SONIC, HuMI, WholeBodyVLA, LeVERB, or GR00T requires a
task with their native observation modality and released components. It would
be a later end-to-end experiment, separate from the current state-only
interface study. At minimum it must preserve:

- the method's visual and language inputs;
- its native command representation and timing;
- its released tokenizer, decoder, action head, and low-level controller;
- comparable demonstrations and task definitions;
- closed-loop success, tracking quality, latency, and hardware assumptions.

Until then, use phrases such as “SONIC-motivated token interface” or
“HuMI-style keypoint packet,” not “SONIC baseline” or “HuMI reproduction.”

## Paper Position and Claims

The literature suggests the following defensible position:

> Recent humanoid systems use both explicit trajectory interfaces and learned
> latent interfaces. SONIC reports that discrete motion tokens are easier for
> its VLA to predict than explicit SMPL poses, while HuMI shows that an overly
> sparse end-effector interface can lose essential whole-body intent. We test
> the interface question under a stricter controlled protocol: a continuous
> latent skill code versus the complete explicit command packet consumed by a
> strong frozen tracker, with identical causal planner inputs, data, model
> families, and closed-loop evaluation.

Our likely contribution is therefore not “the first latent action model.” It
is a controlled measurement of when a learned low-level skill interface makes
high-level whole-body planning easier, including data efficiency, planner-size
scaling, bandwidth, latency, and oracle-normalized closed-loop tracking.

The result remains useful if the explicit baseline catches up at larger model
sizes. That would support a narrower and more credible claim: the latent
interface reduces the planner capacity or data needed for useful control,
while explicit full-body generation retains a strong high-capacity ceiling.

For an ICLR submission the paper must additionally be positioned inside the
representation-learning lineage: spectral dynamics factorization
(SPEDER, LV-Rep, Diff-SR), forward-backward behavior foundation models
(Meta Motivo), predictability-based skill discovery (DADS), hierarchical
interface theory (Nachum et al., 2018), and JEPA-family decoder-free world
models. The precise claim to defend is that a decoder-free spectral encoding
of skill-conditioned macro-dynamics, deployed as the command of a frozen
tracker, reduces the planner capacity and data needed for closed-loop
whole-body control — and the representation-objective ablation, not only the
latent-versus-explicit comparison, is what carries the "spectral" part of
that claim. If the structured-latent composition study matures, it enters as
one secondary section, never as the headline result.

## Primary Sources

- SONIC: <https://arxiv.org/abs/2511.07820>
- SONIC project: <https://nvlabs.github.io/GEAR-SONIC/>
- GR00T Whole-Body Control code:
  <https://github.com/NVlabs/GR00T-WholeBodyControl>
- HuMI: <https://arxiv.org/abs/2602.06643>
- WholeBodyVLA: <https://arxiv.org/abs/2512.11047>
- WholeBodyVLA code: <https://github.com/OpenDriveLab/WholebodyVLA>
- LeVERB: <https://arxiv.org/abs/2506.13751>
- SENTINEL: <https://arxiv.org/abs/2511.19236>
- LangWBC: <https://arxiv.org/abs/2504.21738>
- Humanoid-VLA: <https://arxiv.org/abs/2502.14795>
- GR00T N1: <https://arxiv.org/abs/2503.14734>
- LAPA: <https://arxiv.org/abs/2410.11758>
- Diffusion Policy: <https://arxiv.org/abs/2303.04137>
- ACT: <https://arxiv.org/abs/2304.13705>
- pi0: <https://arxiv.org/abs/2410.24164>
- ASE: <https://arxiv.org/abs/2205.01906>
- CALM: <https://arxiv.org/abs/2305.02195>
- H-GAP: <https://arxiv.org/abs/2312.02682>
- HOVER: <https://arxiv.org/abs/2410.21229>

Representation-learning lineage:

- SPEDER: <https://arxiv.org/abs/2208.09515>
- LV-Rep: <https://arxiv.org/abs/2212.08765>
- Diff-SR: <https://arxiv.org/abs/2406.16121>
- Spectral Representation-based RL: <https://arxiv.org/abs/2512.15036>
- FB representations: <https://arxiv.org/abs/2103.07945>
- Does Zero-Shot RL Exist?: <https://arxiv.org/abs/2209.14935>
- Meta Motivo / FB-CPR: <https://arxiv.org/abs/2504.11054>
- BFM-Zero: <https://arxiv.org/abs/2511.04131>
- BFM position paper: <https://arxiv.org/abs/2506.20487>
- Soft FB: <https://arxiv.org/abs/2602.06769>
- FB critical re-examination: <https://arxiv.org/abs/2602.11399>
- DADS: <https://arxiv.org/abs/1907.01657>
- METRA: <https://arxiv.org/abs/2310.08887>
- HILP: <https://arxiv.org/abs/2402.15567>
- Near-optimal HRL representations: <https://arxiv.org/abs/1810.01257>
- I-JEPA: <https://arxiv.org/abs/2301.08243>
- TD-JEPA: <https://arxiv.org/abs/2510.00739>
- DINO-WM: <https://arxiv.org/abs/2411.04983>
- LAPO: <https://arxiv.org/abs/2312.10812>
- VQ-BeT: <https://arxiv.org/abs/2403.03181>
- QueST: <https://arxiv.org/abs/2407.15840>

Physics-character skill latents and explicit planners:

- NPMP: <https://arxiv.org/abs/1811.11711>
- ControlVAE: <https://arxiv.org/abs/2210.06063>
- PHC: <https://arxiv.org/abs/2305.06456>
- PULSE: <https://arxiv.org/abs/2310.04582>
- MaskedMimic: <https://arxiv.org/abs/2409.14393>
- PADL: <https://arxiv.org/abs/2301.13868>
- FLD: <https://arxiv.org/abs/2402.13820>
- Hybrid part-wise latents: <https://arxiv.org/abs/2503.12814>
- InsActor: <https://arxiv.org/abs/2312.17135>
- BeyondMimic: <https://arxiv.org/abs/2508.08241>
- Diffuse-CLoC: <https://arxiv.org/abs/2503.11801>

When adding a paper, record whether its latent is used only for pretraining or
is actually sent to a low-level controller at deployment. This single check
prevents the most common misleading comparison in this area.
