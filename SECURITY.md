# Security policy

## Supported versions

Security fixes target the latest release on `main`.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting or security advisory feature for
this repository. Include:

- affected version or commit;
- reproduction steps;
- expected and observed behavior;
- impact and any suggested mitigation.

The maintainer will acknowledge a complete report, investigate it, and
coordinate disclosure after a fix is available.

## Scope

Vanitas-Harness launches local model processes and provides path-constrained
file tools. Treat `bin/`, `models/`, model files, prompts, and local workspace
contents as untrusted inputs. Model weights and upstream binaries are not
distributed by this repository.
