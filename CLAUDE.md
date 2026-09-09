@AGENTS.md

# Claude Code sessions: delegate to Codex by default

Applies to Claude Code only. Codex reads `AGENTS.md`, not this file, so nothing
here reaches a delegated agent.

Codex is the default worker in this repo. Claude Code is the interface: it decides
what to ask for, checks what came back, and talks to the user. Assume any unit of
work — reading code, pulling context to answer a question, editing, running
commands — goes to Codex through `mcp__codex__codex`, and justify the exception
when it does not. Follow the `delegate-to-codex` skill for prompts, output
contracts, call parameters, and verification; read it before the first delegation
of a session.

Claude Code keeps only:

- Trivial shell it already knows the exact command for and whose output is a few
  lines: `git status --short`, `git log --oneline -5`, `ls`, `wc -l`, a `sed -n`
  of a known line range.
- Work Codex cannot do or cannot validate: anything under `pixi run -e isaaclab`,
  a GPU, cluster submission, or a call on experimental protocol
  (`AGENTS.md`, "Experiments and results").
- Actions that need the user's approval to surface, and the user-facing answer.
- A single edit to a file already in context, where the round trip costs more
  than the edit.

Do not read a file to build a Codex prompt, and do not re-read what Codex
reports — verify with `git diff --stat` and a test exit code instead. When
Codex's answer is enough, relay it; do not re-derive it locally.
