# doctools-m8

Media-neutral document toolkit: models and protocols, revision identities,
the versioned and integrity-checked artifact envelope with a bounded
incremental reader, the PDF and plain-text parsers, the parser sandbox
supervisor (invocations, private execution tree, attestation) and the
provenance verifier for the sandbox-attestation and gateway-receipt chain.

> Status: scaffold. The package imports and exposes `__version__` only; the
> models, envelope, parsers, sandbox supervisor and verifier land in later
> releases.

## Install

```bash
pip install doctools-m8
# PDF parsing
pip install "doctools-m8[pdf]"
# the parser sandbox supervisor
pip install "doctools-m8[sandbox]"
# the provenance verifier
pip install "doctools-m8[provenance]"
```

The core package depends on Pydantic v2 only, and `media-service-m8` installs
only the core. No network, database, credential or media dependency is
carried anywhere in this package.

## Development

```bash
pip install -e ".[test,dev,pdf]"
ruff format --check . && ruff check .
mypy doctools_m8
bandit -r doctools_m8 --severity-level medium
pytest --cov-fail-under=100
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
