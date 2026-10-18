"""CI, release-workflow, licensing and static template policy checks."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PIPY = WORKFLOWS / "PiPy.yml"
_USES = re.compile(r"uses:\s+([\w.-]+/[\w.-]+)@(\S+)")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def test_every_action_is_pinned_by_full_sha() -> None:
    """Third-party actions are pinned to a 40-hex commit SHA."""
    refs = [m for p in WORKFLOWS.glob("*") for m in _USES.findall(p.read_text())]
    assert refs
    assert all(_SHA.match(ref) for _, ref in refs)


def test_workflows_default_to_read_only_permissions() -> None:
    """Every workflow declares workflow-level ``contents: read``."""
    for path in WORKFLOWS.glob("*"):
        assert _load(path)["permissions"] == {"contents": "read"}, path.name


def test_release_workflow_has_no_pypi_token() -> None:
    """Trusted Publishing only: no token input anywhere."""
    text = PIPY.read_text(encoding="utf-8")
    assert "PYPI_API_TOKEN" not in text
    assert "password:" not in text


def test_release_publish_job_uses_oidc_and_environment() -> None:
    """Only the publish job gets ``id-token: write`` and a named environment."""
    jobs = _load(PIPY)["jobs"]
    assert jobs["pypi-publish"]["permissions"]["id-token"] == "write"
    assert jobs["pypi-publish"]["environment"]["name"] == "pypi"
    assert "id-token" not in jobs["release-build"].get("permissions", {})


def test_release_workflow_triggers_on_published_release_only() -> None:
    """The release workflow runs on ``release: published`` and nothing else."""
    wf: dict[Any, Any] = _load(PIPY)
    triggers = wf.get("on") or wf.get(True)
    assert triggers == {"release": {"types": ["published"]}}


def test_release_workflow_attaches_spdx_sbom() -> None:
    """An SPDX JSON SBOM is attached to the release."""
    text = PIPY.read_text(encoding="utf-8")
    assert "anchore/sbom-action" in text
    assert "format: spdx-json" in text
    assert "upload-release-assets: true" in text


def test_markdownlint_keeps_md024_siblings_only() -> None:
    """The fleet MD024 override is present."""
    cfg = _load(ROOT / ".markdownlint.yaml")
    assert cfg["MD024"] == {"siblings_only": True}


def test_license_is_unmodified_apache_two_oh() -> None:
    """LICENSE is the stock Apache-2.0 text, no MIT remnant anywhere."""
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "MIT" not in text


def test_pyproject_declares_spdx_apache_license() -> None:
    """``pyproject.toml`` uses the SPDX expression, not a classifier (PEP 639)."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in text
    assert 'license-files = ["LICENSE"]' in text
    assert "License ::" not in text


def test_built_wheel_carries_apache_license_metadata(tmp_path: Path) -> None:
    """The built wheel's METADATA names the SPDX license and license file."""
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=ROOT,
        check=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    result = subprocess.run(
        [sys.executable, "-m", "zipfile", "-e", str(wheel), str(tmp_path / "unpacked")],
        check=True,
    )
    del result
    metadata_dir = next((tmp_path / "unpacked").glob("*.dist-info"))
    metadata = (metadata_dir / "METADATA").read_text(encoding="utf-8")
    assert "License-Expression: Apache-2.0" in metadata
    assert "License-File: LICENSE" in metadata
    assert "MIT" not in metadata
    assert (metadata_dir / "licenses" / "LICENSE").exists()


def test_no_notice_file_present() -> None:
    """No third-party code is vendored, so no NOTICE file is carried."""
    assert not (ROOT / "NOTICE").exists()


def test_tracked_text_files_follow_template_rules() -> None:
    """UTF-8 without BOM, LF, final newline, no workspace paths."""
    paths = [
        *ROOT.glob("*.md"),
        *ROOT.glob("*.toml"),
        *ROOT.glob("*.yml"),
        *ROOT.glob("*.yaml"),
        *WORKFLOWS.glob("*"),
        *(ROOT / ".github").glob("*.yml"),
        *ROOT.glob(".*.yml"),
        *ROOT.glob(".*.yaml"),
        *ROOT.glob(".*.toml"),
    ]
    forbidden = ("/workspace", "fa-workspace", "E:\\Dev", ".workspace/")
    for path in paths:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), path
        assert b"\r" not in raw, path
        assert raw.endswith(b"\n"), path
        text = raw.decode("utf-8")
        if path.name in {"AGENTS.md", "CLAUDE.md", "REPOSITORY_CONTEXT.md"}:
            continue
        assert not any(f in text for f in forbidden), path
