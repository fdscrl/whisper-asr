# CLAUDE.md

Project guidance for Claude Code working in this repository.

## Language policy

- **All documentation is written in English.** This includes `README.md`, `CHANGELOG.md`,
  everything under `docs/`, `memory-bank/`, code comments, docstrings, and any new file
  added to the repository.
- **All commit messages are written in English.** No exceptions, regardless of the
  language used in the conversation.
- Chat replies to the user may be in the user's language; artifacts committed to the
  repository may not.

## Commit convention

Commits MUST follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

```
<type>[optional scope][optional !]: <description>

[optional body]

[optional footer(s)]
```

Rules:

- `type` is one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
  `ci`, `chore`, `revert`.
- `scope` is optional and names the affected area, e.g. `feat(gateway):`, `fix(asr):`,
  `chore(docker):`.
- `description` is in the imperative mood, lower case, no trailing period, and kept
  under 72 characters — "add worker restart signal", not "Added worker restart signal.".
- Breaking changes are marked with `!` after the type/scope **and** a
  `BREAKING CHANGE: <explanation>` footer.
- The body explains *why* the change was made, wrapped at 72 columns, separated from
  the subject by a blank line.
- One logical change per commit. Do not mix unrelated edits.

Examples:

```
fix(asr): bound align-model cache to prevent memory growth

feat(gateway)!: require bearer token on every endpoint

BREAKING CHANGE: unauthenticated clients now receive 401.
```

Commit or push only when the user explicitly asks.

## Operational constraint

Never edit configuration directly on the production host. Changes go through this
repository and are applied by deployment; runtime commands such as
`docker compose up -d` are fine.
