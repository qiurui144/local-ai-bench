# Security Policy

## Supported Versions

This repository is developed on `main`. Security fixes are applied to the
current branch unless a release branch is explicitly announced.

## Reporting a Vulnerability

Do not open a public issue for secrets, credential exposure, or exploitable
remote-execution behavior. Report privately to the repository maintainer.

Please include:

- affected commit or release;
- files, commands, or configuration needed to reproduce;
- observed impact;
- whether credentials, private datasets, or device IPs were exposed.

## Secret Handling Rules

- Never commit `.env`, API keys, SSH passwords, private keys, real device
  credentials, or private dataset material.
- Use `*.env` variables documented in `.env.example`, `targets.yaml`, and
  `docs/deploy-targets.md`.
- Keep real VLM images and user data outside git unless they are explicitly
  approved synthetic fixtures.
- Treat generated benchmark outputs under `output/` as local artifacts unless a
  report has been reviewed and copied into `reports/`.
