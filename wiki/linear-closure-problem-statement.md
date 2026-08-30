# Linear Closure of Skill Latents

*A self-contained problem statement, written to collect outside opinions. The
setup is generic skill-conditioned motion tracking; numbers, robot, and
datasets are omitted deliberately — the question is architectural.*

## 1. Setting

We train a humanoid motion-tracking stack with two levels.

- A **skill encoder** `E` maps a short future motion chunk — the next `H = 10`
  reference states, each ~380 dimensions of root-relative pose — to a latent
  **skill vector** `z ∈ R^256`.
- A **low-level policy** `pi(a | s, z)` runs at 50 Hz and receives a fresh `z`
  every control step.

At deployment a planner emits `z`; the encoder is frozen. So `z`-space is
literally the *action space of the high-level MDP*: whatever geometry it has
is the interface the planner must work in.

The encoder is pretrained on a large mocap corpus with three ingredients:

| ingredient | form |
|---|---|
| generative grounding | a diffusion decoder models `p(s_{t+H} \| s_t, z)` — and in the strongest variant, the full next chunk — so `z` must carry what the future looks like |
| bilinear score | `phi(s, z) = g(z)^T F(s) / sqrt(E)`, used inside energy / next-chunk-prediction losses; `g`, `F` are tanh-bounded MLPs |
| isotropy regularizer | a sketched test pushing the marginal of `z` toward an isotropic Gaussian; optionally a final LayerNorm, which places every `z` on the sphere of radius `sqrt(256)` |

## 2. The property we want

Let `Z_valid = E(natural chunks)` be the set of latents the encoder ever
emits. We want the skill space to be **closed under linear combination**:

> **Desired property (linear closure).** For `z_1, z_2 ∈ Z_valid` and
> reasonable weights (convex, or bounded linear), the combination
> `z_mix = a·z_1 + b·z_2` should be
> **(a) executable** — the frozen policy conditioned on `z_mix` remains
> stable, low-jerk, non-degenerate; and
> **(b) lawful** — the grounded meaning of `z_mix` is a predictable function
> of the meanings of `z_1` and `z_2`.

Motivation: a planner that can interpolate and compose skills algebraically
has a far easier emission problem; skill arithmetic also gives cheap behavior
blending and a cleaner story for generalizing outside the training corpus.

## 3. Why the current structure does not give it

Three separate obstructions:

1. **The score is not linear in `z`.** With `g` an MLP,
   `phi(s, a·z_1 + b·z_2)` has no relation to
   `a·phi(s, z_1) + b·phi(s, z_2)`. The bilinear *form* is there, but the
   nonlinearity of `g` discards its algebra.
2. **The interior is never trained.** Every training `z` is an encoder output
   of a real chunk. Nothing certifies points between them — the decoder and
   the policy may behave arbitrarily on chords of `Z_valid`.
3. **Sphere geometry.** With the final LayerNorm, `||z|| = sqrt(256)` always,
   so straight-line interpolation exits the data manifold by construction;
   only spherical interpolation stays on it. Any linear-closure design must
   either drop the norm constraint or accept spherical rather than Euclidean
   algebra.

## 4. Candidate mechanisms, inside the bilinear family

**Hard linear head.** Set `g(z) = G·z`. Then `phi(s, ·)` is exactly linear:
the set of score functions `{s -> phi(s, z)}` becomes a genuine linear space,
closed by construction, and combining latents combines their score
landscapes: `E_mix = a·E_1 + b·E_2`. Applied to the pairwise chunk-transition
energy (both heads linear), this is a spectral factorization of the
transition operator, with `z` as coordinates in the factor basis. The concern
is capacity: how much prediction quality does a linear head give up?

**Relaxed linearity (the version we lean toward).** Keep `g` an MLP and
penalize its deviation from linearity *only along chords between real
skills*:

```
L_lin = lambda * E_{z1, z2, a} || g(a·z1 + (1-a)·z2) - a·g(z1) - (1-a)·g(z2) ||^2
```

`lambda -> inf` recovers linear behavior on the data span; `lambda = 0` is
the current model; the unpenalized value of the gap is a free diagnostic of
how nonlinear the trained `g` already is. An alternative relaxation splits
`g(z) = G·z + h(z)` and penalizes `||h||`, giving an explicit linear part
with a budgeted correction and a closure-error bound.

**Interior coverage.** Independently of the head, train the diffusion decoder
on mixed conditioning: sample pairs, condition on `z_mix`, and take targets
from the alpha-mixture of the two chunks' futures. Diffusion represents
multimodal targets natively, so this directly teaches
`p(· | z_mix) = mixture` — certifying the interior rather than hoping for it.

**Evaluation.** An alpha-sweep probe: encode chunk pairs, condition the
frozen policy on interpolations (both straight-line and spherical), and
measure survival, jerk, and — the harder part — whether the executed motion
is a lawful blend.

## 5. Questions we want opinions on

- **Q1.** Is closure at the *grounding* level (linear score / decoder
  honoring mixtures) sufficient for closure at the *policy* level, or must
  the policy itself be regularized to respond smoothly in `z` (e.g., a
  Lipschitz penalty on `∂a/∂z`)?
- **Q2.** Which algebra is the right target? Linear score combination gives
  energy *addition* — an intersection-like compromise of the two skills —
  whereas mixture-target training gives distributional *averaging*. These
  disagree; which one should `a·z_1 + b·z_2` mean?
- **Q3.** What is the capacity price of a (near-)linear `g`, and where does
  it bite first — prediction error of the grounding, or downstream tracking
  quality?
- **Q4.** Sphere versus linear space: drop the LayerNorm to make Euclidean
  combination meaningful, or keep it and develop the algebra in spherical
  terms?
- **Q5.** How should *lawfulness* of a mixed skill be measured, beyond "the
  robot survives and moves smoothly"? Is there a good quantitative test that
  the blend is the intended one?
- **Q6.** Does chord-linearity enforced on sampled pairs generalize to the
  whole convex hull, or does the interior need explicit coverage (the
  mixture-target term) regardless?

---

*Context for the curious: the policy is trained with on-policy RL against
tracking rewards; the encoder is frozen during that phase; the latent is
republished every 20 ms.*
