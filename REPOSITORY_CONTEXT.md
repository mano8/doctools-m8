# doctools-m8

## Layer

Platform (document-processing library).

## Purpose

Provide a media-neutral document toolkit: document models and protocols,
revision identities, a versioned and integrity-checked artifact envelope with a
bounded incremental reader, PDF and plain-text parsers, the parser sandbox
supervisor, the provenance verifier, and planning for OCR, chunking, and
extraction.

The package is published to PyPI as `doctools-m8` and imported as
`doctools_m8`. It is licensed under Apache-2.0.

## Repository boundaries

- The core is pure Python with Pydantic v2 as its only dependency. Native and
  optional parsers live behind extras (`pdf`, `sandbox`, `provenance`,
  `docling`).
- The `sandbox` and `provenance` extras use `ai-sdk-m8[jobcontext]` for signing
  and verification. This library implements no cryptography of its own.
- No network, database, credential, object-storage, or media-service
  dependency anywhere. PostgreSQL, pgvector, Vault, media ACLs, and network
  access are out of scope.
- Where the sandbox runs and where the verifier is called belong to the
  consuming service; this library owns their semantics only.

These are repository architecture constraints. If a requested change appears to
break one, record the decision in the workspace context when that optional
enhancement is available and confirm the exception before proceeding.

## Standalone authority

This file, `pyproject.toml`, repository documentation, and existing CI are the
authoritative local context. A verified nearest workspace may optionally add
launcher-selected policies and tasks; its absence is a successful standalone
condition and does not make a parent workspace necessary.
