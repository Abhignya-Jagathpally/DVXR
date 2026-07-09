# Inherited Claude Code skills

Project-scoped skills vendored into this repo so the DVXR work can plan, build, review, and write up
rigorously and reproducibly (offline, version-pinned, committed).

## Sources

- **addyosmani/agent-skills** (24 skills) — engineering-lifecycle workflows: define/plan/build/verify/
  review/ship (e.g. `spec-driven-development`, `test-driven-development`, `code-review-and-quality`,
  `security-and-hardening`, `git-workflow-and-versioning`). License: `LICENSE.agent-skills`.
- **imbad0202/academic-research-skills** (4 skills) — `deep-research`, `academic-paper`,
  `academic-paper-reviewer`, `academic-pipeline`. Directly supports POW Goal 4 (IEEE conference paper).
  License: `LICENSE.academic-research-skills`.

`karpathy/autoresearch` is NOT a skill; it is vendored under `../../third_party/autoresearch/` as
GPU-only reference material (see that README).

## Note — name overlap

The academic pack's **`deep-research`** skill shares a name with Claude Code's built-in `deep-research`
skill. This project-scoped copy (a 13-agent academic-research pipeline) is intentionally kept under its
upstream name; when triggered here it takes precedence over the built-in. Rename this directory and its
`SKILL.md` `name:` field if you prefer to keep both distinct.
