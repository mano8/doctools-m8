# doctools-m8

Media-neutral document toolkit: models and protocols, revision identities,
the versioned and integrity-checked artifact envelope with a bounded
incremental reader, the PDF and plain-text parsers, the parser sandbox
supervisor (invocations, private execution tree, attestation) and the
provenance verifier for the sandbox-attestation and gateway-receipt chain.

> Status: pre-release. The pure-Python core (models, protocols, revisions and
> the artifact envelope) is in place; the PDF parser, the sandbox supervisor and
> the provenance verifier land next.

## Core modules

- `doctools_m8.models` — `ParsedDocument`, `Page`, `Block`, `Chunk` and
  `ExtractionResult` with per-field `Provenance`. Frozen, strict models that
  refuse unknown fields; chunk kinds are honest (`text`/`unknown` unless a
  parser emitted structural tags).
- `doctools_m8.protocols` — `DocumentParser`, `Chunker` (synchronous) and
  `OcrProvider`, `EmbeddingProvider`, `RerankProvider`, `MetadataExtractor`
  (asynchronous, implemented by the deploying service).
- `doctools_m8.revisions` — revision identities: `h(*parts)` is the first 32
  hex characters of the SHA-256 of the RFC 8785 (JCS) encoding of
  `[parts...]`; `document_sha256` is the full SHA-256 of the JCS document.
- `doctools_m8.artifact` — the versioned `document.v1.json.gz` envelope and
  `chunks.v1.jsonl.gz`. Writers are deterministic (one gzip member, `mtime`
  0, no file name, fixed level, JCS JSON, `produced_at` supplied by the
  caller). `read_envelope` / `read_chunks` verify size and SHA-256 while
  streaming, bound decompression and decoding with `ReaderLimits` (bounded
  `zlib` steps and an incremental decoder whose limits trip before the
  governed token is allocated), then check `schema_version`, the schema, the
  declared counts and the expected identity. Any failure raises
  `ArtifactReadError` with a `code`; no partial data is returned.

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
