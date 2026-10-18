# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-18

### Added

- Repository scaffold: hatchling packaging, empty `doctools_m8` package with
  `__version__`, ruff / mypy strict / bandit / pip-audit / pytest (100%
  coverage) CI, PyPI Trusted Publishing release workflow with an SPDX SBOM
  attached to the GitHub Release, Dependabot, Codacy and gitleaks
  configuration. Licensed Apache-2.0. Core dependencies are `pydantic` only;
  `pdf`, `sandbox`, `provenance` and `docling` extras are declared as
  placeholders for later waves.
