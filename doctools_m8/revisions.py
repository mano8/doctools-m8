"""
C-07 revision identities.

Every revision is ``h(*parts)``: the first 32 hex characters of the SHA-256 of
the JCS encoding (RFC 8785) of the JSON array ``[parts...]``, so a revision is
independent of key order and whitespace in any configuration it covers.
``document_sha256`` is the full SHA-256 of ``JCS(ParsedDocument)``.

``normalization_revision`` is known only after execution: the OCR execution
identity is the one the gateway's ``provider_execution`` returned for the
first successful OCR call of the run, pinned for the whole run
(:func:`pin_execution_identity`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from doctools_m8._jcs import canonicalize
from doctools_m8.models import ParsedDocument

__all__ = [
    "REVISION_HEX_LENGTH",
    "ExecutionIdentityChangedError",
    "chunk_revision",
    "config_hash",
    "document_sha256",
    "embedding_space_id",
    "index_revision",
    "normalization_revision",
    "ocr_execution_identity",
    "parse_revision",
    "pin_execution_identity",
    "revision_hash",
]

REVISION_HEX_LENGTH = 32
"""The length of every revision: a 128-bit SHA-256 prefix."""

_NO_OCR = "no-ocr"

Config = Mapping[str, object]


class ExecutionIdentityChangedError(ValueError):
    """A later OCR call resolved to another execution identity than the pinned one."""

    code = "execution_identity_changed"


def revision_hash(*parts: object) -> str:
    """Return ``h(*parts)``: the 32-hex SHA-256 prefix of ``JCS([parts...])``."""
    return hashlib.sha256(canonicalize(list(parts))).hexdigest()[:REVISION_HEX_LENGTH]


def config_hash(config: Config) -> str:
    """Return ``h(config)`` for a configuration mapping."""
    return revision_hash(dict(config))


def document_sha256(document: ParsedDocument) -> str:
    """Return ``SHA256(JCS(document))``, the normalized content's identity."""
    return hashlib.sha256(canonicalize(document.model_dump(mode="json"))).hexdigest()


def parse_revision(
    source_sha256: str, parser: str, parser_version: str, parser_config: Config
) -> str:
    """Return ``h(source_sha256, parser, parser_version, parser_config)``."""
    return revision_hash(source_sha256, parser, parser_version, dict(parser_config))


def ocr_execution_identity(
    provider_id: str, external_model_id: str, model_revision: str
) -> str:
    """Return ``h(provider_id, external_model_id, model_revision)``."""
    return revision_hash(provider_id, external_model_id, model_revision)


def pin_execution_identity(pinned: str | None, observed: str) -> str:
    """
    Pin the run's OCR execution identity to the first one observed.

    Returns ``observed`` when nothing is pinned yet, the pinned identity when
    they agree, and raises :class:`ExecutionIdentityChangedError` otherwise —
    the run is then restarted.
    """
    if pinned is not None and pinned != observed:
        raise ExecutionIdentityChangedError(
            "OCR execution identity changed during the run"
        )
    return observed


def normalization_revision(
    parse_rev: str,
    document_sha: str,
    *,
    ocr_profile_key: str | None = None,
    ocr_execution: str | None = None,
    ocr_config: Config | None = None,
) -> str:
    """
    Return the normalization revision of a parsed (and maybe OCR'd) document.

    With OCR: ``h(parse_revision, ocr_profile_key, ocr_execution_identity,
    ocr_config, document_sha256)``; without: ``h(parse_revision, "no-ocr",
    document_sha256)``. The three OCR arguments are given together or not at
    all.
    """
    ocr = (ocr_profile_key, ocr_execution, ocr_config)
    if all(part is None for part in ocr):
        return revision_hash(parse_rev, _NO_OCR, document_sha)
    if ocr_profile_key is None or ocr_execution is None or ocr_config is None:
        raise ValueError("the OCR profile, execution identity and config go together")
    return revision_hash(
        parse_rev, ocr_profile_key, ocr_execution, dict(ocr_config), document_sha
    )


def chunk_revision(
    normalization_rev: str,
    chunker: str,
    chunker_version: str,
    tokenizer_version: str,
    chunker_config: Config,
) -> str:
    """Return ``h(normalization_revision, chunker, chunker_version, tokenizer_version, chunker_config)``."""
    return revision_hash(
        normalization_rev,
        chunker,
        chunker_version,
        tokenizer_version,
        dict(chunker_config),
    )


def embedding_space_id(
    provider_id: str,
    external_model_id: str,
    model_revision: str,
    dimensions: int,
    execution_config_hash: str,
) -> str:
    """Return ``h(provider_id, external_model_id, model_revision, dimensions, execution_config_hash)``."""
    return revision_hash(
        provider_id,
        external_model_id,
        model_revision,
        dimensions,
        execution_config_hash,
    )


def index_revision(
    chunk_rev: str, embedding_profile_key: str, embedding_space: str
) -> str:
    """Return ``h(chunk_revision, embedding_profile_key, embedding_space_id)``."""
    return revision_hash(chunk_rev, embedding_profile_key, embedding_space)
