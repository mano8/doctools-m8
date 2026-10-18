# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-18

### Added

- Core models and protocols: `ParsedDocument`, `Page`, `Block`, `Chunk`,
  `ExtractionResult` with per-field `Provenance`; `DocumentParser`,
  `Chunker`, `OcrProvider`, `EmbeddingProvider`, `RerankProvider` and
  `MetadataExtractor`.
- `doctools_m8.revisions`: JCS-canonical SHA-256 revision identities
  (`parse_revision`, `document_sha256`, `ocr_execution_identity`,
  `normalization_revision`, `chunk_revision`, `embedding_space_id`,
  `index_revision`) and the pinned OCR execution identity.
- `doctools_m8.artifact`: the deterministic `document.v1.json.gz` envelope
  and `chunks.v1.jsonl.gz` writers, and their bounded readers — size and
  SHA-256 verified while streaming, one gzip member only, bounded inflate
  steps, a pure-Python incremental JSON decoder whose depth, string, number,
  member, item, key, text, page, block and chunk limits trip before
  allocation, a wall-clock bound, and count and identity checks.
- Repository scaffold: hatchling packaging, empty `doctools_m8` package with
  `__version__`, ruff / mypy strict / bandit / pip-audit / pytest (100%
  coverage) CI, PyPI Trusted Publishing release workflow with an SPDX SBOM
  attached to the GitHub Release, Dependabot, Codacy and gitleaks
  configuration. Licensed Apache-2.0. Core dependencies are `pydantic` only;
  `pdf`, `sandbox`, `provenance` and `docling` extras are declared as
  placeholders for later waves.
