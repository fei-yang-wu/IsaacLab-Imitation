# Affine skill learning and product-of-experts composition

Research review, 2026-09-12. These are theoretical deductions and proposed
experiments, not measured improvements or an approved campaign. The user's
observation that the direct 64-D head performs worse is taken as motivation;
this review does not independently rescore checkpoints or query running jobs.

Implementation follow-up: the user chose the minimal diffusion change and
authorized submission of a base-field arm against the existing PoE control.
`z64_poe_base` was submitted as jobs `5766785 -> 5766786 -> 5766787`.
[Campaign and validation](../experiments/campaigns/2026-09-12-poe-base-z64-10b/README.md).
The clean-EBM and flow proposals below remain unimplemented research ideas.

Follow-up requirement: the user explicitly prioritizes linear composition of
the command itself. Under this requirement, the nonlinear-code and block-swap
recommendations below are comparison ideas, not the preferred design. The
preferred direction is clean-density affinity with unrestricted noisy scores;
see the follow-up at the end of this note.

## Implementation checked

The `2026-08-30-past-chunk-affine-64d` campaign uses deterministic 64-D z,
`source_history_steps=5`, horizon 10, and the merged `diff_chunk` head with
`boundary_next` and endpoint coefficient zero. The prediction target is raw
frames at the boundary and through the following chunk, not an EMA token.
Its feature width is 256 and embedding width is 1024. The source includes
the current state and five past states. Historical README dimension comments
are inconsistent with the later recorded 380-value macro-state contract;
use the checkpoint's resolved dimensions when implementing an experiment.

Code references:

- `RLOpt/rlopt/agent/ipmd/module.py`: `BilinearSR.forward_phi`,
  `DiffSRBilinear.forward_mu`, `forward_eps`, and `compute_loss`.
- `RLOpt/rlopt/agent/hl_skill_diffsr.py`: merged raw-chunk target construction
  and `_ntp_diffsr_loss`.
- `experiments/campaigns/2026-08-30-past-chunk-affine-64d/campaign.yaml`.
- `experiments/campaigns/2026-09-11-poe-z64-10b/README.md`.

Writing y for the future chunk and tau for diffusion time, the code predicts

$$\widehat\epsilon(y_\tau,\tau\mid s,z)
=\phi(s,z)^T M(y_\tau,\tau),\qquad
\phi(s,z)=F(s)^T(Az+b)/\sqrt{1024}.$$

`M` is a directly predicted matrix of vector fields. Despite the Jacobian
terminology in its docstring, it is not computed as the derivative of a
scalar-feature network. The loss is squared noise-prediction error, through
which gradients reach the window encoder. The corresponding score estimate
is `-eps / sqrt(1-alphabar_tau)` in normalized target coordinates.

## What the affine algebra proves

For a scalar log potential (the exponent, using the user's positive sign),

$$f(s,z,y)=\langle F(s)^T(Az+b),\mu(y)\rangle
=c(s,y)+z^T T(s,y),$$

where `c=b^T F(s) mu(y)` and `T=A^T F(s) mu(y)`, absorbing fixed scaling.
Thus the affine head is already a 64-coordinate exponential-family model
with a base potential. Widening internal features does not add independent
command directions: its dependence on z still spans at most 64 directions
plus the offset. A sufficiently expressive joint `T(s,y)` with an offset
can represent this function class. Finite architectures, parameter counts,
factor sharing, normalization and optimization can nevertheless differ.

When normalizers exist, at fixed s, scalar potentials imply

$$p(y\mid s,\sum_i w_i z_i)
\propto p(y\mid s,0)^{1-\sum_iw_i}
\prod_i p(y\mid s,z_i)^{w_i}.$$

For convex weights this is a geometric pool. Arbitrary weights need the
base correction and need not define a normalizable distribution. Latent
coordinates are signed natural parameters, not intrinsically semantic
experts or log probabilities. A product of full-motion distributions also
need not mean executing different motion attributes simultaneously.

The implemented field has the same additive algebra, but a free field need
not be conservative. Furthermore, summing exact scores of separately
noised distributions does not generally give the score obtained by noising
their clean product. [Du et al., ICML 2023](https://proceedings.mlr.press/v202/du23a.html)
analyze this sampling mismatch and develop annealed MCMC approaches;
explicit scalar energies enable Metropolis corrections. Consequently the
older wiki/campaign statements that affine mixtures *sample the exact
product* are too strong. This note qualifies those statements without
changing historical experiment configurations.

There is a second consequence, derived here: requiring affinity at every
noise level is stronger than requiring a clean exponential-family model.
Consider `p(y|z)=N(0,1/lambda(z))`, with positive affine precision lambda.
Its clean log potential is affine in z. Under VP corruption with alpha,

$$\nabla_{y_\tau}\log p_\tau(y_\tau\mid z)
=-\frac{y_\tau}{\alpha/\lambda(z)+(1-\alpha)},$$

which is generally nonlinear in z. Even a valid clean PoE can therefore be
misspecified by the all-noise affine score constraint.

The original [Diff-SR paper](https://arxiv.org/html/2406.16121v1) connects
factorized log potentials to spectral representations through additional
feature transformations. This does not make raw 64-D z a universal linear
value representation, nor prove that a separately trained controller
executes the learned conditional on unseen commands.

## Literature most relevant to the design

| Work | What to borrow | Boundary |
| --- | --- | --- |
| [Du, Li and Mordatch, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/49856ed476ad01fcff881d57e161d73f-Abstract.html) | Add log potentials for conjunction of constraints. | Products require compatible support; they are not trajectory averaging. |
| [COMET, NeurIPS 2021](https://arxiv.org/html/2111.03042v1) | Infer several compact codes and sum nonlinear component energies. | COMET explicitly formulates energies as optimization costs without a probabilistic interpretation. |
| [Decomp Diffusion, ICML 2024](https://arxiv.org/html/2406.19298v1) | Jointly infer low-dimensional component codes; train a sum of conditioned denoisers with one denoising loss. | Image-factor results do not establish controllable humanoid factors or exact PoE sampling. |
| [MCP, NeurIPS 2019](https://arxiv.org/html/1905.09808v1) | Compose Gaussian action policies multiplicatively, with learned nonnegative gates and per-action uncertainty. | A controller-level alternative requiring its own matched comparison. |
| [SPiRL, CoRL 2020](https://arxiv.org/html/2010.11944v1) | A compact skill embedding, nonlinear sequence decoder, and state-conditioned skill prior. The reported embedding has 10 dimensions. | Its tasks and reconstruction demands do not establish that 10 dimensions suffice here. |

The essential distinction is that additivity across experts does not require
linearity within each expert's code. Decomp Diffusion is the closest
architectural precedent for retaining denoising pretraining and a small
command interface.

## Proposed experiments, in order

### 1. Restore a command-independent base in the direct head

Compare the current direct head to

$$\widehat\epsilon=\epsilon_{\rm base}(s,y_\tau,\tau)
+\sum_{j=1}^{64}z_j V_j(s,y_\tau,\tau).$$

The existing homogeneous head predicts zero at z=0, although even the
high-noise Gaussian needs a nonzero score. It must represent common
denoising through the codes themselves. An offset removes that burden
without increasing the planner command. This is a hypothesis about training,
not proof of the observed failure: the encoder might avoid zero or learn a
nearly constant coordinate already.

Match source history, regularizers, data, diffusion schedule, widths/budget,
and tracker recipe. The September 11 package changes multiple variables
relative to older controls. Separate architectural output width from command
dimension when attributing a result.

### 2. Keep 64 command values, make four nonlinear experts

Let `z=(z_1,z_2,z_3,z_4)`, each block in R^16. Proposed log potential:

$$f(s,z,y)=f_{\rm base}(s,y)+\sum_{k=1}^4 f_k(s,y;z_k).$$

Each expert can have a large network, nonlinear in its small code, with
access to the full state/target context. Composition now replaces or combines
expert blocks. Straight-line interpolation of the entire z is no longer
guaranteed to interpolate energies. Exact global Jensen affinity and this
extra nonlinear flexibility cannot both be demanded of the same code map.

A practical Decomp-Diffusion-inspired first implementation retains direct
noise prediction and trains

$$\mathcal L=\mathbb E\left\|\epsilon-
\left[\epsilon_{\rm base}+\sum_k\Delta\epsilon_k(s,y_\tau,\tau;z_k)\right]
\right\|^2.$$

This is an additive denoiser, not a guaranteed scalar EBM. A later explicit
energy variant uses `eps=-sigma_tau*grad_y f_tau`; it guarantees conservative
fields but introduces mixed second derivatives in training and does not
alone repair diffusion-time composition consistency.

Four blocks are a proposed starting point, not a literature optimum. Avoid
claiming they spontaneously mean locomotion, arms, posture and style. Test
factor swaps on matched phase/contact contexts. If semantic independence is
needed, add separately ablated evidence such as partial-body observations,
task attribute supervision, or invariance under augmentations that preserve
the chosen factor. Whole-body coupling must remain available to the model.

### 3. Test nonlinear lifting as the simpler capacity control

Use `h:R^64 -> R^256` (or a matched wider feature space), and

$$f=f_{\rm base}+\langle h(z),T(s,y)\rangle.$$

This gives a curved family of log potentials with 64 interface values. It
is distinct from affine widening `Az+b`, which adds no independent linear
directions. It preserves neither global latent arithmetic nor closure of
the lifted manifold under feature addition. Recovering a compact z for a
sum of lifted codes is an approximation unless separately established.
Existing nonlinear bilinear/concat infrastructure offers useful controls.

### 4. Make the bottleneck predictive and executable

Dimension alone is not an information-rate constraint on real-valued codes.
If needed, separately test a stochastic posterior with a conditional prior:

$$\mathcal L=\mathcal L_{\rm pred}
+\beta\mathbb E\,D_{\rm KL}\big(q(z\mid s,w)\Vert p(z\mid s)\big).$$

This KL upper-bounds `I(z;w|s)` and encourages the code to convey information
not supplied by history. This conditional-rate proposal is inspired by skill
prior work, not a claim that it is SPiRL's exact encoder objective. A strong
decoder can ignore z, so monitor shuffled-code degradation and control
sensitivity. Tune beta separately rather than bundling regularizer changes.

Also distinguish predictive information from command-relevant information:
different intermediate paths may share the same future chunk. The current
objective alone does not require preserving everything a tracker needs to
execute the encoded interval. A modest executed-chunk or teacher-action
auxiliary is a separate proposed ablation, not a reason to replace the
predictive loss wholesale. Future windows remain offline labels; the planner
receives only causal history and explicit task input.

For stronger execution semantics, MCP offers a concrete controller:
`pi(a|s,z) proportional to product_k pi_k(a|s)^{w_k(s,z)}`. Gaussian expert
precisions and precision-weighted means add, giving tractable sampling.
Otherwise train the existing tracker on validated composed references/codes;
pretraining algebra by itself does not constrain closed-loop execution.

## Evaluation and claim boundary

First compare held-out noise error by noise level and motion feature group,
code shuffling, code usage and norm, and compatibility of factor swaps.
Use compatible states and phases. Evaluate conjunction with separate
attribute errors (for example, root velocity and hand target), not closeness
to an average of incompatible reference motions. For explicit energies,
small synthetic distributions can check integrability and composition before
expensive humanoid tests.

Then compare matched tracker SR, MPJPE-L and MPJPE-G together, command-rate
and physical jitter, and planner success at the intended publication cadence.
Use matched frames and repeated seeds for conclusions. A compact command
at every control step and one held across a planner interval are different
bandwidth contracts; keep cadence fixed in the head comparison.

Recommended next steps: biased direct-head diagnostic, then a four-by-16
nonlinear expert model with a matched single-decoder nonlinear-lift control.
All remain proposals; no training or implementation was performed here.

## Follow-up: retain linear command composition

The desired constraint belongs on the clean conditional log potential:

$$p_\theta(y\mid s,z)=\exp\{c_\theta(s,y)+z^TT_\theta(s,y)-\mathcal A_\theta(s,z)\}.$$

Here z remains 64-D. Both c and T may have large nonlinear networks. Their
scalar output makes the clean score conservative, and convex combinations
of commands give exact normalized geometric pools whenever the component
densities are normalizable. The flow of noisy distributions need not remain
affine: its log density is the logarithm of an integral of the clean density
against a Gaussian kernel. That integration is the source of the nonlinearity.

For a general clean EBM, likelihood training has gradient
`-grad f(s,z,y_positive) + E_p[grad f(s,z,y_negative)]`, including gradients
through the window encoder's z. Model negatives must hold that positive
window's (s,z) fixed. Finite MCMC gives an approximate training gradient.
Normalizability needs a suitable base/tail constraint or bounded domain.
An alternative is [score matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf),
with [sliced score matching](https://proceedings.mlr.press/v115/song20a.html)
reducing derivative computation through random projections. Standard
consistency results require smooth densities and boundary conditions;
constrained motion coordinates and a jointly learned encoder require care.

A nonlinear diffusion head can remain an auxiliary predictor or proposal
generator. An independently trained head is not automatically consistent
with the clean EBM. To sample the latter, use target-aware MCMC correction
or train a sampler against it and quantify approximation error. A residual
whose coefficient vanishes at zero noise only enforces an endpoint condition;
it does not establish the correct family of noised marginals.

A tractable alternative, derived for this task using the change-of-variables
construction of [normalizing flows](https://jmlr.org/papers/v22/19-1028.html), is

$$u=g_\theta(y;s),\qquad
p(y\mid s,z)=\mathcal N(u;m(s)+B(s)z,\Sigma(s))\,|\det J_y g_\theta|.$$

Require g to be invertible and independent of z, and Sigma to be positive
definite and independent of z. Expanding the quadratic gives
`c(s,y)+z^T B(s)^T Sigma(s)^(-1)(g(y;s)-m(s))`, plus terms independent of y.
It therefore belongs exactly to the desired clean exponential family,
although g and the resulting motion samples are nonlinear. Likelihood and
sampling are tractable, with `y=g_inverse(m+Bz+L*xi;s)`. Train the window
encoder jointly by conditional negative log likelihood and an independently
controlled latent regularizer. A noninvertible decoder or a z-conditioned
flow does not inherit this proof. Real motion constraints require an
appropriate continuous coordinate/density model.

This fixed-covariance Gaussian family in transformed coordinates is more
restrictive than an arbitrary clean EBM. It is a useful concrete baseline
for exact linear composition, not a guarantee of improved tracking. It
retains a 64-dimensional family of log-density changes; large networks
increase feature complexity, not the number of independent command
directions. An affine precision parameter can extend the family on a domain
where the precision stays positive definite, as a separate experiment.

Linear composition still needs a specified semantic operation. Convex
combinations are geometric pooling. Relative addition around z_ref gives
`p(y|s,z_A+z_B-z_ref) proportional to p_A*p_B/p_ref` when normalizable,
avoiding duplication of the reference behavior. All identities hold at the
same s. They do not guarantee semantic disentanglement or dynamically
feasible execution by the separately trained policy. Composed-reference
training and held-out attribute conjunction tests remain necessary.

## September 12 follow-up: joint PoE versus structured affine decoder

Read the recent Claude session `40ffd517-fef5-487b-8b73-629eb5440f1c`
and checked its 1,000,243,200-frame rows against
`logs/latent64_probe_live_eval/*_seed0_clean_f1000243200.json`.
The following are seed-0, early-training comparisons with regularizers on,
past-five source history, and matched tracker settings. MPJPE is measured
on successful trajectories, whose population changes across arms.

| Arm | SR | MPJPE-L mm | MPJPE-G mm |
|---|---:|---:|---:|
| p5_affine_ctrl | 0.8572 | 29.29 | 185.6 |
| z64_poe_reg | 0.7947 | 35.73 | 269.8 |
| z64_poe_base_reg | 0.6714 | 48.26 | 499.8 |
| poe_proj256_reg | 0.8096 | 34.22 | 246.8 |
| poe_proj256_base_reg | 0.8130 | 35.10 | 251.6 |

These rows support an early gap, not a final-convergence or causal conclusion.
`enc_hist` is a nonlinear concat head, not the affine control. The original
combo tracker has a different protocol and should not substitute for this
control. Adding the independent base has not established a benefit.

The implementation predicts epsilon vector fields; it does not construct
scalar energies or enforce conservative scores. Write the joint field as
`D(s,y_t,t)` to avoid confusing it with a scalar energy. With fixed trunk h,
ResidualMLP ends in a plain linear output, so
`D = D_0 + sum_j h_j(s,y_t,t) D_j`. The configured last hidden width is 512.
A fixed linear latent lift A can be absorbed into each output matrix D_j:
`(Az)^T D = z^T (A^T D)`. Consequently, widening the projected expert bank
does not enlarge this unrestricted final-layer function class at fixed h;
it changes optimization and parameterization. An affine lift similarly
folds into an effective latent field and base field.

The original affine head instead supplies explicit products between learned
source features and learned noisy-target features:
`epsilon = (Az+b)^T F(s) M(y_t,t) / sqrt(embed_dim)`.
An arbitrary joint field can express this abstractly, but the particular
finite joint MLP is not guaranteed to contain the factored architecture.
Its output matrix is also not uniquely large: both the structured control
and projected-256 arms have a 256-by-target-dimension target-field output.
Loss of multiplicative structure is a stronger architectural hypothesis
than matrix size alone. We have not established the causal mechanism.

Recommended next work, not submitted: retain the existing affine decoder
as the main compositional baseline. First fit joint and structured decoders
with the same frozen affine encoder, examples, and noise draws; inspect
validation loss by noise level and the singular values of the command
Jacobian. This separates decoder fit from encoder co-adaptation. A fresh
matched affine/next versus affine/pair pretrain isolates adding source
input to mu while retaining F. Do these offline screens before another
tracker-scale campaign.

If a smaller structured arm is wanted, directly predict B(s) and d(s) and
use `epsilon = [B(s)z+d(s)]^T M(y_t,t)`, with B of shape 256-by-64.
The old head maps into this through B=F^T A/sqrt(embed) and
d=F^T b/sqrt(embed). This retains affine command dependence and explicit
source/target multiplication, but finite network parameterizations and
optimization differ; improvement is untested. Affinity still guarantees
only the field composition identity at each noise level, not exact clean
PoE sampling or semantic composition by the downstream policy.

## Pretrain-only rescue campaign

The user requested a pretrain-only z64 screen, then asked for more modeling
and representation innovation. The resulting
[rescue campaign](../experiments/campaigns/2026-09-12-poe-rescue-pretrain/README.md)
contains fresh affine/next, affine/pair, direct joint PoE, a wider joint head,
no-bias factored PoE, and source-modulated PoE. Every arm has 50k updates and
matched regularizers/data; no tracker is submitted.

The source-modulated proposal keeps phi=z and uses
`E = reshape(W[(1+gamma(s))*h(y_t,t)+beta(s)] + b)`.
The products are between source and noisy-target features; z only weights the
resulting 64 expert fields. This moves expressivity into the state/transition
representation without nonlinearizing or widening the planner command.
It avoids the original large state matrix, but does not claim to contain its
finite function class. This adaptation uses established
[FiLM](https://arxiv.org/abs/1709.07871) and
[bilinear-pooling](https://arxiv.org/abs/1610.04325) ideas; research novelty and
benefit remain unestablished.

Representation-wise, linearity at a fixed history is weaker than shared
semantic axes across histories. A useful subsequent question is whether the
same z direction consistently controls a motion attribute across source
states; held-out attribute conjunctions or structured body/task components
would test that. Dynamics loss alone tests achievable predictive fit, not
that semantic claim. A further hypothesis is tension between centered codes
and a strictly linear denoiser's required high-noise base field. A fixed
Gaussian reference score, leaving <z,E> as a log-density-ratio model, is a
possible follow-up; it is not included in this first screen and would change
the reference measure interpretation of the pure exponent.

Qualification also found an existing trainer checkpoint reload omission for
jepa_state_dict. Restoring the active NTP head is now tested, so future
post-hoc dynamics evaluations do not silently use freshly initialized heads.
Fresh training's in-process metrics were not affected by that reload bug.
