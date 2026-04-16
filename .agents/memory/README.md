# Repo memory

This repository uses Pi-managed project-local memory under `.agents/memory/`.

## Layout

- `core/` — four short markdown blocks (`directives.md`, `context.md`, `focus.md`, `pending.md`). Their combined total must stay at or below 300 lines and 20000 characters. Enforce both caps only when writing core or finalizing a dream, not when reading.
- `log.md` — append-only markdown log for decisions, prompts, plans, experiments, attachment additions, and lessons from failed or rejected attempts. Use explicit `supersedes` and `invalidates` links when an entry replaces or corrects prior memory.
- `compactions/` — immutable dream and compaction summaries with provenance.
- `research/` — use it for short abstracts of actual external SOTA research that materially informs the current work. Read relevant files on demand before relying on them.
- `attachments/` — only user-requested or user-facing files used to collaborate with the user. This folder is gitignored. Do not use it as scratch space. If you add something here, also append a log entry naming the file and why it exists.

## Write rules

- If work is left unfinished, update `core/pending.md`.
- If architecture or behavior changes, update `core/context.md`.
- If user preferences or standing constraints change, update `core/directives.md`.
- If focus shifts, update `core/focus.md`.
- If an important decision, prompt ingest, plan, experiment, attachment addition, or lesson from a failed or rejected attempt happens, append to `log.md`.
- If a new log entry replaces or corrects prior memory, include explicit `supersedes` and/or `invalidates` links.
- If the user provides a substantial brief, append it as a `prompt | high` log entry.

## Dream

Dream is the only consolidation mechanism for `core/`.

Pi can trigger dreaming automatically on session start when logs are stale and after compaction when new compaction context should be folded into memory. You can also run `/memory dream` manually.

Dream reads all log entries newer than the last dreamed log timestamp plus pending compaction summaries. It does not automatically retrieve older log entries.
