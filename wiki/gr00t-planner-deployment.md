# GR00T planner: asynchronous eval and deployment design

Status: batched paper-evaluation design plus a local native deployment path,
2026-08-11. The training pipeline is built and smoke-tested locally. The
single-robot path is implemented in `external/Embodied-Control`: a C++ 50 Hz
tracker publishes 10x93 causal histories through a shared-memory request
slot, a non-real-time worker calls `gr00t_chunk_service`, and the returned
30x38 chunk reaches a second slot. The service sends this chunk once; its
duplicate 10x38 encoder-window view is optional diagnostic data. C++ then
runs the DiffSR encoder, policy,
and fake/MuJoCo or optional Unitree backend. The batched Isaac construction
below remains a design unless marked DONE.

## Definitions

- **Planner service**: one process that owns the trained GR00T action head
  and produces command chunks. It runs in the `gr00t` Pixi environment.
- **Sim runtime**: for paper-scale evaluation, one process owns Isaac Lab.
  For local deployment tests, the C++ Embodied-Control runtime owns MuJoCo
  and the frozen 50 Hz tracker without Isaac Lab.
- **Host orchestrator**: `external/Embodied-Control` (`ec eval run`,
  delegated mode). It launches the two processes, waits, and normalizes
  artifacts.
- **RTC**: real-time chunking. The planner freezes the actions that will
  execute during inference and inpaints the rest, conditioned on the
  previous chunk tail. The verbatim head already implements this
  (`get_action(..., options={rtc_*})`; frozen-slot preservation is verified
  by our smoke test).

## Why two processes

1. **Version isolation.** The head trains under torch 2.9.0 (upstream pin);
   Isaac Lab owns torch 2.11. The stub import works today, but a process
   boundary removes the coupling permanently.
2. **True asynchrony.** The planner GPU work overlaps simulator stepping.
   In-process, a 1.3B-head forward blocks the control loop; across
   processes it hides behind the 10 control steps of the current hold.
3. **It is the Embodied-Control contract.** EC's whole design is "policy
   and simulator never import each other" — delegated mode, host
   networking (Apptainer-safe on Skynet), one artifact contract
   (`episodes.jsonl`, `metrics.json`, `manifest.json`, `status.json`).

## Transport

The implemented single-robot path uses two fixed shared-memory seqlock slots
and a line-JSON stdio child process. The control tick only reads a slot; it
never calls Python. ZeroMQ + msgpack remains the planned batched transport
(GR00T's native server transport; `pyzmq` is already in
the upstream pin set). `ipc://` socket on one machine; `tcp://127.0.0.1`
under containers with host networking. EC's HTTP/JSON transport stays what
it is — a debug transport; the GR00T client adapter presents the same
`health/describe/reset/act` surface to EC.

The service emits a ready record before requests are accepted. The native
worker requires the exact deployment contract: 10 state frames, 93 state
values per frame, 38 action values per frame, and the bundle's encoder-window
length. A wrong response sequence, non-finite chunk, or shape mismatch is a
fault. The two shared-memory slots each have exactly one owner.

Payload sizes at 4096 envs: state history 10x93 f32 = 3.7 KB/env, chunk
30x38 f32 = 4.6 KB/env. A full-fleet renewal round-trip is ~15 MB up /
~19 MB down — trivial for ipc, fine for local tcp.

## The request loop (sim side)

The existing `FrozenContinuousInterfacePlannerSampler` semantics stay: per-env
hold countdown, renewal on `done | steps<=0`, no global modulo. Two changes:

1. **Lead-time request.** For env `i`, at `lead` control steps before its
   hold expires, enqueue a chunk request with its current causal state and
   goal id. `lead` = measured planner latency in control steps, rounded up.
   This is the RTC `frozen` region.
2. **Swap at expiry.** At expiry, the response chunk is already in the
   per-env double buffer; swap it in. If it is late (deadline miss), hold
   the previous chunk's next slots and count the miss — never block the
   control loop, never fabricate. When the late result is used, select its
   encoder window with the actual ticks elapsed since its matching request.
   Do not replay the originally planned lead-time frames.

Requests are batched: each control step, all envs whose request timer fired
form one batch to the service. Asynchronous resets simply put an env's
request in a later batch — the per-env timers already handle it.

## The service loop (planner side)

- Loads: head checkpoint (+ its normalization stats and provenance),
  per-goal cached Cosmos text features (preloaded to GPU — ~100 goals is
  megabytes), RTC options.
- Serves: `act(states, goal_ids, prev_chunk_tails) -> chunks`, one batched
  `get_action` call with the RTC options and the previous tails inpainted.
- Optimizations, in order of expected payoff:
  1. bf16 inference (weights + activations; ~2.6 GB for the 1.3B head).
  2. `torch.compile` on the 4-step denoise loop (fixed shapes after the
     first batch; pad the batch dimension to a small set of bucket sizes).
  3. Pinned-memory staging buffers for the zmq <-> GPU copies.
  4. CUDA graphs only if compile alone misses the budget.
- Latency accounting: CUDA-synchronized timestamps around the root forward
  only (the frozen protocol's definition), plus wall-clock request-to-reply
  for the deadline-miss statistic. Both go into `metrics.json`.

## Eval construction

Phase the work; each phase is independently verifiable:

- **Phase D0 (correctness baseline, in-process).** Stub-import the head in
  the isaaclab env (DONE for import + forward), run the existing closed-loop
  evaluator with synchronous inference. Slow but exact; the reference the
  async path is compared against.
- **Phase D1 (service + async client, no EC).** The zmq service and the
  batched async sampler, both launched by one wrapper script. Gate (relaxed
  2026-08-18 by user decision — the earlier match-within-eval-noise
  certificate was judged too strong a requirement): a D0 comparison run on
  the same seeds and goals is REPORTED next to the async numbers, with the
  gap and the deadline-miss statistic stated, but no numeric equivalence
  bound blocks D2 or an async row. An async row must always be labelled
  async and cite its D0 companion; the two are never pooled.
- **Phase D2 (EC delegated orchestration).** EC launches both runtimes and
  owns artifacts. Container images: `gr00t` env (service) and the existing
  Isaac image (sim). Skynet uses Apptainer + host networking, which EC
  already targets. This is the shape for paper-scale batch eval.

## What stays ours regardless

Per-env asynchronous renewal, anchor re-expression for explicit packets,
termination/metric semantics, checkpoint binding and provenance gates, and
the diagnostic full-horizon pass with video. The service replaces only the
"where does the chunk come from" edge of the sampler.

## Open items

- Gated `nvidia/Cosmos-Reason2-2B` access (user action) blocks the real
  goal-feature cache; the warm-start bundle path does not need it.
- Stage-B (trunk-unfrozen) training exceeds 20 GB locally; it is a cluster
  stage. Local machine covers stage A, debug presets, D0, and D1 at small
  env counts.
- Deadline-miss policy under load at 4096 envs (hold vs. skip-one-slot)
  needs a measurement before choosing.
- RTC overlap is off by default in the local native worker. On the
  debug-scale stage-A head, lead-4 without RTC survived, while RTC compounded
  chunk-tail error and fell. Re-enable it only after a full checkpoint passes
  a matched stability test.
