# Security

`doctools-m8` is a media-neutral document toolkit. The core package holds no
network, database or credential dependency; it never calls out and reads no
settings or environment variables. The `[sandbox]` extra runs untrusted parser
output inside a supervised, custody-tracked process boundary — see the
project's sandbox invocation contract for its isolation guarantees.

## Supply chain

- Releases are published to PyPI only through Trusted Publishing (OIDC); no
  upload token is stored anywhere.
- Every release attaches a Software Package Data Exchange (SPDX) JSON software
  bill of materials (SBOM) to its GitHub Release.
- CI actions are pinned by full Secure Hash Algorithm (SHA) commit hash and
  run with read-only default permissions.
- `pip-audit` and `bandit` run on every pull request.

## Reporting a vulnerability

Report security vulnerabilities privately through GitHub's **Security** tab on
this repository — **"Report a vulnerability"** — which opens a private
security advisory visible only to the maintainers. Do not open a public GitHub
issue for vulnerabilities. Expected response within 48 h.
