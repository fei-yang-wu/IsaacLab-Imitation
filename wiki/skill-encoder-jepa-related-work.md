# Related Work for the Unified Skill Encoder Plan

Last reviewed: 2026-08-13. Companion to
[skill-encoder-jepa-plan.md](skill-encoder-jepa-plan.md); method-level
comparison only. Sources marked **(snippet)** were assessed from search
summaries, not read; verify before citing in a paper. Everything else was read
at least at abstract level. The broader interface literature lives in
[whole-body-vla-literature-review.md](whole-body-vla-literature-review.md);
this page covers only what the JEPA/chunk proposal touches.

## The proposal, restated for comparison

One encoder maps a fixed-length chunk of proprioceptive motion (10 macro
slots) to a code. The same encoder serves three roles: the code of the
preceding chunk is the context, the code of the executed chunk is the
"action", and the code of the following chunk is the prediction target. The
predictor is trained JEPA-style in code space with SIGReg as the anti-collapse
term; no policy gradient, no reconstruction in the loss (reconstruction stays
as a diagnostic). Data is proprioceptive chunks from both mocap references and
the robot's own rollouts. Deployment is unchanged: the code is a closed-loop
~5 Hz command consumed by a frozen RL tracker.

The user's framing is accurate: structurally this is (a) an autoregressive
model over skill codes — next-token prediction at chunk granularity — and
(b) a latent action model where the "action" is a chunk code. The families
below are organized around that.

## A. Video latent action models (LAM family)

LAPO and [Genie] began the line in video games; robotics followed:
[LAPA](https://arxiv.org/abs/2410.11758) (latent actions from human/robot
video, then finetune to real actions), IGOR (human-to-robot latent transfer),
Moto-GPT (co-finetunes latent and real action labels), GR00T N1 (latent
actions as an extra "embodiment"), GO-1 (actions conditioned on discrete
latent tokens), UniVLA (task-centric latents),
[villa-X](https://arxiv.org/abs/2507.23682) (adds a proprioceptive
forward-dynamics model to ground the latents). 2026 continuations include a
hierarchical LAM ([2603.05815](https://arxiv.org/pdf/2603.05815) (snippet)),
depth-aware latents (UniLACT (snippet)), latent distillation (LatBot
(snippet)), and [VLA-JEPA](https://arxiv.org/pdf/2602.10098) (snippet), which
bolts a latent world model onto a VLA.

**Method-wise difference.** A LAM's latent action is defined between two
adjacent observations — an inverse-dynamics bottleneck: infer the latent from
(o_t, o_{t+1}) such that a forward model can produce o_{t+1}. Our code is the
embedding of the chunk itself: a temporal abstraction of states, not a
frame-pair residual. Second, LAM latents are scaffolding — they pretrain a
VLA and are then decoded away into explicit robot actions. Our code **is** the
deployed interface; nothing decodes it away. Third, their substrate is video;
ours is proprioception. The overlap is the *idea* that action = learned latent
between observations; the mechanism, granularity, and deployment differ at
every step.

**What to borrow.** villa-X's observation that ungrounded latents drift into
shortcut features (camera motion, lighting) is the video version of our
copy-baseline worry; their fix — a proprioceptive forward model grounding the
latent — is structurally our predictor.

## B. Next-token prediction over trajectory/skill codes

The closest structural relatives.
[TAP](https://arxiv.org/abs/2208.10291) encodes multi-step
(state, action, reward, return) segments into discrete codes with a
state-conditioned VQ-VAE and learns an autoregressive Transformer prior
p(z_i | z_<i, s_1); [H-GAP](https://arxiv.org/pdf/2312.02682) is the humanoid
instantiation (MoCapAct, 56-DoF), planning over the code prior.
[I-TAP](https://arxiv.org/html/2602.18694) (Feb 2026, concurrent) stacks
residual-quantized tokens over observation-macro-action segments,
autoregressively predicts token stacks from recent history, and runs MCTS in
token space. [PRISE](https://arxiv.org/abs/2402.10450) quantizes per-step
actions then applies BPE to get variable-length skills; VQ-BeT and QueST
learn discrete action-chunk vocabularies for BC;
[FAST](https://arxiv.org/abs/2501.09747) tokenizes action chunks in frequency
space for autoregressive VLAs; the T2M-GPT/MotionGPT lineage does next-token
over kinematic motion tokens without physics.

**Method-wise difference, per axis:**

- *Code training objective.* All of these train the tokenizer by
  reconstruction (VQ-VAE or DCT); the prior is a second model trained after or
  alongside. Ours replaces reconstruction with latent-target prediction
  (JEPA + SIGReg), and the "prior" — predict the next code — **is** the
  representation objective, not a separate stage. Whether that is an
  improvement is exactly what the plan's phase 2 vs phase 3 comparison
  measures; the literature does not answer it.
- *What the token contains.* TAP/H-GAP embed reward and return in the token;
  our code deliberately contains no reward — task selection lives in the
  planner, not the skill space.
- *Executor.* TAP/H-GAP/I-TAP decode tokens to raw actions and execute them
  open-loop (H-GAP notes MPC-style replanning); VQ-BeT/QueST/FAST decode to
  action chunks inside a BC policy. Ours never decodes: a frozen
  physics-trained RL tracker consumes the code closed-loop at 50 Hz. That
  executor absorbs dynamics error the open-loop decoders must model, which is
  the main reason our code can be smaller than their token stacks.
- *Discrete vs continuous.* This family is uniformly discrete. Our plan keeps
  the code continuous (z256) or FSQ-quantized (fsq64) as separate arms; the
  Discrete Codebook World Models result
  ([2503.00653](https://arxiv.org/pdf/2503.00653) (snippet)) — discretizing a
  latent world-model state improved sample efficiency over continuous — is
  weak evidence the fsq64 arm matters here too.

## C. JEPA-family predictive models and the anti-collapse toolbox

[I-JEPA](https://arxiv.org/abs/2301.08243) set the template: predict
embeddings, not pixels.
[V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) post-trains an
action-conditioned latent dynamics model on ~62 h of robot data and plans
zero-shot on Franka arms — the strongest evidence that planning in a
predictive embedding space works on real robots. DINO-WM plans in a frozen
feature space; TD-JEPA (2510.00739) connects latent-predictive learning to
successor features; Navigation World Models do the same for locomotion in
video space.

**Anti-collapse.** [LeJEPA (2511.08544)](https://www.alphaxiv.org/overview/2511.08544v1)
proves the isotropic Gaussian is the optimal embedding prior for downstream
risk and enforces it with
[SIGReg](https://www.emergentmind.com/topics/sketched-isotropic-gaussian-regularization-sigreg):
random 1-D projections + an Epps–Pulley characteristic-function test, linear
cost, one trade-off weight, no teacher/EMA/stop-gradient. Two caveats found in
review: the regularization signal *weakens as the embedding approaches
collapse* (it detects non-Gaussianity, and a near-constant embedding plus
noise can pass 1-D tests before full collapse), so it should be paired with
the effective-rank and per-dimension-variance gates already in the plan; and
[VISReg (2606.02572)](https://arxiv.org/html/2606.02572v1) (June 2026,
concurrent, (snippet)) proposes variance-invariance-sketching as an
alternative — worth one read before freezing the choice.

**The control-side sibling.** [TD-MPC2](https://arxiv.org/pdf/2310.16828) is
the closest *mechanism*: a latent dynamics model trained by latent-consistency
regression, explicitly no reconstruction, collapse prevented by SimNorm
(group-wise softmax onto simplices) plus grounding through reward/value
heads. Differences: TD-MPC2's action is the *real* per-step action and its
latents are grounded by reward; our "action" is a code from the same encoder
(self-conditioned on both sides, hence the stronger anti-collapse need), our
grounding is SIGReg + the copy-baseline gate, and our granularity is the
chunk. SimNorm is a cheap fallback if SIGReg underdelivers: it is
architecture-level, not loss-level, and composes with everything else in the
plan.

**V-JEPA 2-AC vs ours in one line:** they condition a video-embedding
predictor on real actions to get a plannable model; we condition a
chunk-embedding predictor on a code from the same space to get a plannable
*skill* model. The self-conditioning is the novelty and the risk.

## D. Humanoid deployed latent interfaces (system-level concurrent work)

- **SONIC** (2511.07820): the reference point. Encoder inside the PPO actor,
  FSQ tokens, reconstruction auxiliary at 0.01, PG flows through the encoder,
  no dynamics model over tokens, cross-embodiment diversity from human/SMPL
  modes. Our plan is the opposite corner: no PG, dynamics model over codes,
  diversity from the robot's own rollouts.
- **[ω-0 (2608.06375)](https://arxiv.org/html/2608.06375)** (Aug 2026,
  directly concurrent): "latent predictive world-action model" for humanoid
  loco-manipulation. Predicts compact future *observation* embeddings as a
  lightweight auxiliary (JEPA-spirited foresight), generates
  controller-compatible whole-body action latents via diffusion, grounds
  human/video motion priors through SONIC-based simulation replay, 40+ h real
  household dataset. Differences: its foresight is visual and auxiliary, its
  action latents come from a *frozen upstream tokenizer* (SONIC), and there is
  no dynamics model *in the action-latent space*. It validates the
  latent-command-to-tracker deployment at system scale but does not test our
  representation question. **Must be cited as concurrent work.**
- **MetaWorld (2601.17507)** (Jan 2026 (snippet)): hierarchical world model,
  VLM semantic layer over a latent dynamics model in compact state space,
  expert-policy library as motion priors. Adjacent at the architecture level;
  does not learn the skill code and its dynamics with one encoder.
- **WholebodyVLA** (ICLR 2026, OpenDriveLab (snippet)): "unified latent VLA
  for whole-body loco-manipulation" — triage before citing; likely the
  LAM-for-humanoid corner.
- **LeVERB, PULSE, MaskedMimic, R2S2** (prior work): CVAE/distillation skill
  latents consumed by learned controllers; all reconstruction-trained, none
  with a code-space dynamics model, none with the one-encoder unification.
- **DreamPolicy (2505.18780)** (snippet): unified world-model policy for
  humanoid locomotion; triage.

## E. Diffusion language models and diffusion-forcing sequence models

Added at the user's request 2026-08-13: how the plan relates to diffusion LMs.

The diffusion-LM line (LLaDA, masked/score discrete diffusion; survey
[2508.10875](https://arxiv.org/pdf/2508.10875) (snippet)) replaces
left-to-right next-token decoding with iterative parallel denoising of masked
token sequences. It has already reached action decoding:
[LLaDA-VLA (2509.06932)](https://arxiv.org/abs/2509.06932) finetunes a
diffusion VLM to discretized action tokens with localized special-token
classification and hierarchical decoding, and reports beating AR VLA
baselines; [Discrete Diffusion VLA (2508.20072)](https://arxiv.org/html/2508.20072v4)
puts discrete-diffusion action decoding inside one transformer;
MMaDA-VLA ([2603.25406](https://arxiv.org/pdf/2603.25406) (snippet)) and
Fast-dVLA ([2603.25661](https://arxiv.org/pdf/2603.25661) (snippet)) are the
2026 native/pretrained and real-time continuations.
[Diffusion Forcing (2407.01392)](https://arxiv.org/abs/2407.01392) sits
between the camps: per-token independent noise levels unify next-token
prediction and full-sequence diffusion in one causal model, giving
variable-horizon rollout, guidance toward rewards/goals, and the useful prior
that the far future is more uncertain than the near future.

**Where this actually touches the plan.** These works answer a different
question than JEPA + SIGReg. The representation question — what the chunk code
is and why it does not collapse — is untouched by all of them: diffusion VLAs
consume tokens someone else defined (discretized raw actions, FAST bins). The
sequence-model question — given codes, how to model p(next codes | history) —
is exactly what they answer, and there the plan currently assumes the simplest
choice (a one-step regressor/denoiser). Three concrete implications:

1. **A deterministic predictor averages multimodal futures.** After a chunk
   ends mid-stance, several continuations are plausible; a single-point
   code-space regressor learns their mean, which may be no valid continuation.
   A denoising predictor over the next code (the DiffSR machinery moved into
   code space) or a masked-diffusion model over code sequences represents the
   multimodality instead. Our own stack already leans this way — DiffSR is a
   denoiser and the GR00T planner head is flow-matching — so a diffusion
   predictor in code space is the consistent choice, not an exotic one.
2. **The fsq64 arm makes the discrete-diffusion analogy literal.** 64
   dimensions x 32 levels is a token sequence; masked discrete diffusion over
   chunk codes is then structurally identical to a small diffusion LM whose
   vocabulary is the skill lattice. Any-order decoding buys goal-conditioned
   *infilling*: clamp the current chunk code and a goal chunk code, denoise
   the codes between them — plan synthesis in skill space, which an AR prior
   can only do by search (compare I-TAP's MCTS).
3. **Diffusion Forcing is the right template if the predictor becomes a
   planner.** Training the code-sequence model with per-chunk noise levels
   gives stable rollout past the training horizon and guidance in code space,
   with causality preserved — the properties H-GAP/TAP get from AR priors plus
   search, in one model.

None of this changes phases 0-3 of the plan; it is the natural phase-4 shape
if the phase-2/3 code space proves informative. The anti-collapse story must
still come from the representation objective — a diffusion decoder on top of a
collapsed code space would simply model noise fluently.

## F. The unification lineage (state repr = skill repr)

The "collapse state representation and skill representation into one" idea has
a specific ancestry: METRA (skill = direction in a learned state-embedding
space, trained so embedding displacement aligns with the skill vector), HILP
(Hilbert state representation; skills are directions in it), FB / Meta Motivo
(task vector and successor representation share one space), TD-JEPA (ties the
two formally through latent-predictive TD). All of these unify at the
*per-state* level with online or offline RL objectives. Ours unifies at the
*chunk* level with a pure prediction objective and no reward — closest in
spirit to what METRA would look like if its state embedding were a trajectory
embedding and its directional constraint were replaced by a next-chunk
predictor. This is the right family to position against in a paper's
related-work section, because the unification claim is otherwise easy to
overstate: **per-state versions of it exist; the chunk-level, deployed-
interface version is what is new.**

## Comparison table

| method | code = | trained by | anti-collapse | dynamics over codes | executor | data |
| --- | --- | --- | --- | --- | --- | --- |
| ours (plan) | chunk embedding | next-code prediction (JEPA) | SIGReg (+rank/variance gates) | yes — the objective itself | frozen RL tracker, closed loop | mocap + own rollouts, proprio |
| SONIC | FSQ token of ref window | PPO through encoder + recon aux | recon + PG grounding | no | RL tracker (same net) | mocap + human/SMPL |
| TAP / H-GAP / I-TAP | VQ/RVQ token of (s,a,r) segment | VQ-VAE reconstruction | codebook | separate AR prior | decode to actions, open loop | offline RL / MoCapAct |
| PRISE / VQ-BeT / QueST / FAST | quantized action chunk | reconstruction | codebook | AR prior (BC head) | decode to actions | demos |
| LAM family (LAPA…villa-X) | frame-pair inverse-dynamics latent | forward-model reconstruction | bottleneck + grounding heads | no (villa-X adds FDM) | decoded away after pretraining | video |
| V-JEPA 2-AC | video patch embeddings | latent-target prediction | EMA teacher + masking | conditioned on real actions | MPC over model | video + 62 h robot |
| TD-MPC2 | per-step state latent | latent consistency + reward/value | SimNorm + reward grounding | conditioned on real actions | MPC + policy | online RL |
| METRA / HILP / FB | state embedding; skill = direction/task vector | RL/spectral objectives | objective structure | implicit | learned policy | online/offline RL |
| ω-0 | SONIC token (frozen) + visual foresight embedding | diffusion head + embedding prediction aux | inherited from SONIC | no (visual only) | SONIC tracker | 40 h real household |
| diffusion VLAs (LLaDA-VLA, DDVLA, MMaDA-VLA) | discretized raw-action tokens | masked-diffusion denoising | n/a (tokens given) | no — decodes actions, not dynamics | decoded actions, open-loop chunk | demos |
| Diffusion Forcing | any continuous token | per-token-noise denoising | n/a (tokens given) | yes — causal, guidable | task-dependent | video / planning |

## What is genuinely different, and what is not

Not novel on its own: predicting embeddings instead of raw targets (JEPA);
autoregression over skill tokens (TAP/H-GAP); latent commands to a tracker
(SONIC/LeVERB); chunk-level abstraction (ACT onward); SIGReg (LeJEPA).

The specific untested combination: **the deployed command code, the state
representation, and the dynamics model's state are the same object, trained
only by next-chunk prediction in that space, with no reward, no policy
gradient, and no reconstruction in the loss, executed closed-loop by a frozen
physics RL tracker, fit on the robot's own physically-feasible rollouts.**
Every neighbor above lacks at least two of those properties. The two nearest
threats to that claim are I-TAP (concurrent; discrete, reconstruction-trained,
open-loop decode, but otherwise the same "prior = dynamics over segment
tokens" shape) and ω-0 (concurrent; same deployment story, no representation
claim). Both must appear in any writeup.

Risks the literature explicitly warns about, mapped to plan gates: shortcut
latents (villa-X grounding discussion) → copy baseline on tiled chunks;
SIGReg's weakening signal near collapse → effective-rank and per-dimension
variance gates; reconstruction-free latents drifting away from motion
semantics (TD-MPC2 grounds via reward, which we refuse) → reconstruction kept
as a diagnostic gate, never a loss.

## Triage before citing

Unread beyond abstract/snippet: VISReg, I-TAP details, MetaWorld,
WholebodyVLA, DreamPolicy, UniLACT, LatBot, VLA-JEPA, hierarchical LAM
(2603.05815), Discrete Codebook World Models, ω-0 full method section
(abstract only), MMaDA-VLA, Fast-dVLA, the diffusion-LM survey (2508.10875).
LeJEPA's exact Epps–Pulley statistic, projection count, and batch-size
interaction still need the primary read flagged in the plan.
