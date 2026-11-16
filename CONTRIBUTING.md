# Contributing

## Workflow

1. Create a short-lived branch from `main`.
2. Keep one logical change per commit and use Conventional Commits.
3. Add tests for behavior changes.
4. Run the verification commands below before opening a pull request.

## Environment

Vanitas-Harness currently targets Windows and Python 3.13.

```powershell
python -m pip install -r requirements.txt
Get-ChildItem vanitas -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
Get-ChildItem tests -Recurse -Filter test_*.py | ForEach-Object { python -m py_compile $_.FullName }
python scripts\check_comments.py
python -m unittest discover -s tests
```

## Repository rules

- Do not commit `bin/`, `models/`, generated reports, logs, credentials, or
  local absolute paths.
- Only `prompt_toolkit` is allowed as a third-party runtime dependency.
- Public benchmark data must be original synthetic material. Do not add source
  or translated subtitles unless you own the rights and document the grant.
- Do not hand-edit subtitle bodies in place of fixing prompts, glossaries,
  deterministic post-processing, or tests.
- Never use `git clean -fdx`; ignored model and binary directories may be lost.
- Preserve the public CLI, configuration schema, and AgentLoop contracts unless
  a breaking change is explicitly documented.

## Pull requests

Describe the behavior change, risks, verification performed, and rollback
path. State explicitly if real-model or hardware-dependent behavior was not
verified.
