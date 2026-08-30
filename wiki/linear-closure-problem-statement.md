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
   nonlinearity of `g` discards its algebra. Measured on the production
   encoder over 1,000 cross-motion pairs, the midpoint gap
   `||phi(s, z_mid) - mean(phi(s,z_1), phi(s,z_2))||` is 11.1% of the mixed
   score's norm on average and 47.3% at worst — the obstruction is real, not
   a formality.
2. **The interior is never trained.** Every training `z` is an encoder output
   of a real chunk. Nothing certifies points between them — the decoder and
   the policy may behave arbitrarily on chords of `Z_valid`.
3. **The interior may be off-manifold.** Straight-line interpolation could
   leave the region the encoder ever populates, in which case the decoder and
   the policy see inputs they were never trained on.

   *Correction, 2026-08-30.* An earlier version of this document claimed the
   encoder's final LayerNorm places every `z` on the sphere of radius
   `sqrt(256)`, making chords off-manifold by construction. That is wrong: the
   encoder's output layer is a bare linear map and the layer-norm option gates
   hidden layers only, so `z` lives in unconstrained `R^256` with only SIGReg
   and a weak L2 pulling on its marginal. Measured on 1,000 cross-motion pairs
   from the production encoder, the midpoint of a chord has 0.72x the norm of
   a real latent and sits 1.05x the real nearest-neighbour distance from the
   real set — near the manifold, not off it. There is no convexity obstruction
   and no need for spherical algebra.

## 4. The chosen mechanism: an affine head

**Decided 2026-08-30.** One design is being built. The chord-penalty
relaxation, the `g(z) = G·z + h(z)` residual split, and mixture-target
interior coverage that earlier drafts of this document proposed are dropped,
not deferred.

Set `g(z) = A·z + b`, keeping `F(s)`, `mu`, and the encoder as nonlinear as
they are today. Then

```
phi(s, z) = F(s)^T (A z + b) / sqrt(E)
```

is affine in `z`, and because the diffusion head predicts noise as
`eps_pred = phi(s,z)^T mu(y_t, t)` with `mu` blind to `z`, the entire learned
score field of `p(y | s, z)` is affine in `z` at every diffusion time. For
weights summing to one, the score of the mixed latent is the mixture of the
scores — the score of the geometric mixture:

```
p(y | s, z_a)  proportional to  p(y | s, z_1)^a p(y | s, z_2)^(1-a)
```

This settles **Q2**: a linear combination of latents means the *product* of
the endpoint conditionals, an intersection-like compromise, not a
distributional average. `A` and `b` are learned; the constraint is on the
functional form only. `A` is identifiable only up to an invertible linear map
(`z -> Rz`, `A -> AR^-1` gives identical scores), which is harmless — linear
reparameterization preserves affine combinations — but it means individual
coordinates of `z` are not interpretable.

Written in operator form, `M(z) = M_0 + sum_k z_k M_k` with
`f(s, y, z) = psi(y)^T M(z) phi(s)`: each coordinate of `z` linearly weights
one learned transition pattern, and `M_0` is the bias `b`. The matrix is never
materialized; the code computes the vector `A z + b` and contracts it.

**Cost, and where it is measured.** The open concern is capacity (**Q3**): how
much prediction quality a linear `g` gives up. That is measured directly by
the pretraining eval losses of an affine encoder against a matched control,
before any tracker is trained.

**Evaluation.** Two stages. Offline, `probe_latent_interpolation` measures the
score-affinity gap, the geometry of the interpolant against the real-`z` set,
and how the endpoint head's denoising error trades off across an alpha sweep.
In simulation, once a tracker exists, an alpha-sweep probe conditions the
frozen policy on interpolations and measures survival, jerk, and whether the
executed motion is a lawful blend. Only the second stage can answer **Q1**;
the affine head constrains the grounding, and the policy is a separate
network with no linearity constraint of its own.

## 5. Questions we want opinions on

- **Q1.** Is closure at the *grounding* level (linear score / decoder
  honoring mixtures) sufficient for closure at the *policy* level, or must
  the policy itself be regularized to respond smoothly in `z` (e.g., a
  Lipschitz penalty on `∂a/∂z`)?
- **Q2.** ~~Which algebra is the right target?~~ **Answered by the chosen
  design (section 4): the product, `p_1^a p_2^(1-a)`.** An affine score gives
  geometric mixture semantics exactly, and mixture-target training — which
  would have given distributional averaging — is dropped, so the two no longer
  compete.
- **Q3.** What is the capacity price of a (near-)linear `g`, and where does
  it bite first — prediction error of the grounding, or downstream tracking
  quality? *Open; this is what the campaign measures.*
- **Q4.** ~~Sphere versus linear space?~~ **Moot: there is no sphere** (see
  the correction in section 3). `z` is already unconstrained, so Euclidean
  combination is the natural algebra and straight-line interpolation is the
  probe's default.
- **Q5.** How should *lawfulness* of a mixed skill be measured, beyond "the
  robot survives and moves smoothly"? Is there a good quantitative test that
  the blend is the intended one? *Partially addressed offline: the denoising
  error of the mixed latent against each endpoint's own true future should
  trade off smoothly across the sweep. The in-sim analogue is still open.*
- **Q6.** ~~Does chord-linearity enforced on sampled pairs generalize to the
  whole convex hull?~~ **Moot for the chosen design:** an affine `g` is exact
  on the whole space, not just on sampled chords, so there is no
  generalization gap to worry about. What remains open is whether the encoder
  *populates* the interior, which is the geometry measurement in section 4.

---

*Context for the curious: the policy is trained with on-policy RL against
tracking rewards; the encoder is frozen during that phase; the latent is
republished every 20 ms.*
