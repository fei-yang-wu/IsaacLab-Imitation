---
name: delegate-to-codex
description: Route work to Codex via the codex MCP server — the default for reading code, pulling context, answering questions, editing, and running commands in this repo. Use whenever a task needs more than a trivial known one-liner. Codex keeps only Isaac-env/GPU/cluster work, protocol judgment, approvals, and the user-facing answer.
---

# Delegating to Codex

The `codex` MCP server runs a full Codex agent out-of-process. Its exploration and
generation bill to the user's ChatGPT plan, not Anthropic's — but everything Codex
*returns* enters this conversation permanently and is re-sent on every later turn.
Delegation is the default; protecting that return channel is what makes it pay.

## Routing

Delegate by default. Send Codex:

- **Questions about the code** — "how does X work", "where is Y set", "what does
  this config control". Do not open files to answer; ask Codex.
- **Context pulls** — locating call sites, tracing a value through modules,
  summarizing a subsystem or a campaign directory.
- **Edits** — one file or fifty, new modules with tests, mechanical refactors,
  doc sweeps, config changes.
- **Commands** — builds, greps that need iteration, `pixi run test-experiments`,
  `pixi run test-rlopt`, `pixi run test-scripts`, log triage, anything whose
  output would be long or whose exact invocation is not already known.

Codex keeps:

- Trivial shell with a known exact command and a few lines of output:
  `git status --short`, `git log --oneline -5`, `ls`, `wc -l`, `git diff --stat`,
  a `sed -n` of a known line range.
- Work Codex cannot validate: `pixi run -e isaaclab ...`, GPU runs, cluster
  submission, and any call on experimental protocol, budget, or comparison.
- Anything needing user approval to surface — MCP calls run with
  `approval-policy: never`, so a destructive or outward-facing action must not
  be inside a delegation.
- A single edit to a file already in context.
- The synthesis and the user-facing answer. Codex reports; Codex decides.

## Before delegating

Check the working tree with `git status --short`. If it is dirty in the area
Codex will touch, say so and ask whether to commit, stash, or use a worktree — a
clean tree is what makes `git diff --stat` a trustworthy record of what Codex did.
For read-only work a dirty tree is fine.

## Building the prompt

**Do not read files to write the prompt.** Reading them here pays the exploration
cost this skill exists to avoid, then pays again to transmit it. Give Codex paths
and intent; it reads the repo itself.

Do not restate repo conventions. Codex reads the root `AGENTS.md` natively, so the
Pixi rule, ownership boundaries, and test suites arrive for free.

State the task, the paths or entry points if known, and what "done" means. Then end
every prompt with the matching output contract, verbatim.

**Edits and commands:**

> Final message: at most 10 lines. List each file changed with one line on what
> changed, and the exit status of any command you ran. No code blocks, no diffs,
> no account of your reasoning. If you are blocked, name the file and the
> ambiguity and stop.

**Questions and context pulls:**

> Final message: at most 15 lines. Answer the question directly and cite
> `path:line` for each claim. No code blocks longer than 3 lines, no file dumps,
> no account of how you searched. Say "not found" rather than guessing.

## Calling it

`mcp__codex__codex` with:

- `prompt` — task, paths, and the matching output contract
- `cwd` — the repository root (`imitation_experiments.paths.REPO_ROOT`)
- `sandbox` — `"read-only"` for questions and context pulls;
  `"workspace-write"` for edits and commands
- `approval-policy` — `"never"`
- `model` — `"gpt-6-astra"`
- `config` — `{"model_reasoning_effort": "low"}`; raise to `"medium"` for design
  work or a diagnosis that has already failed once at low

Without the sandbox and approval settings it stalls on approvals it cannot surface
through MCP.

Iterate with `mcp__codex__codex-reply` on the returned thread id — never a second
`codex` call, which forces re-authoring the whole brief here. Keep one thread per
work item and follow up in it; the thread holds the exploration Codex never paid
for. `codex-reply` inherits the thread's model, so it needs no override.

Independent work items can be dispatched as separate `codex` calls in one message.

## Verifying

For edits: `git diff --stat` plus a test exit code. Read actual code only when a
stat line is wrong — an unexpected file, an implausible line count. Reading the
full diff by reflex forfeits the savings.

Run the gate yourself when Codex cannot: `pixi run -e isaaclab test-isaaclab`,
smoke runs, anything needing the Isaac env or a GPU. Otherwise let Codex run the
suite and report the exit status, and spot-check only when the claim looks off.

For answers: trust a cited `path:line` unless it matters enough to be worth a
one-line `sed -n` — a wrong answer about protocol, a contract in `AGENTS.md`, or a
number that will reach the wiki is worth the check.

## Reporting

Relay what changed, what the tests did, and what Codex answered. If Codex was
blocked, say so plainly rather than papering over it, and do not silently take the
work back — either re-delegate with the ambiguity resolved or tell the user why
this one belongs in Codex.
