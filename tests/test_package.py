"""Package import, version parity and core-dependency boundary checks."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import doctools_m8

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_and_exposes_version() -> None:
    """The empty package imports and exports only ``__version__``."""
    assert doctools_m8.__all__ == ["__version__"]


def test_version_matches_pyproject() -> None:
    """``__version__`` equals the pyproject literal."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == doctools_m8.__version__


def test_changelog_has_current_version_heading() -> None:
    """CHANGELOG.md carries a Keep a Changelog heading for the version."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = (
        rf"^## \[{re.escape(doctools_m8.__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
    )
    assert re.search(pattern, text, re.MULTILINE)


def test_core_dependencies_are_pydantic_only() -> None:
    """The core wheel depends on pydantic alone."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[<>=!~ ]", d)[0] for d in data["project"]["dependencies"]}
    assert names == {"pydantic"}


def test_extras_match_the_scaffold_contract() -> None:
    """Each extra carries exactly the dependency set T7 ratifies."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    def names(extra: str) -> set[str]:
        return {re.split(r"[<>=!~ ]", d)[0] for d in extras[extra]}

    assert names("pdf") == {"pypdfium2"}
    assert extras["sandbox"] == ["ai-sdk-m8[jobcontext]"]
    assert extras["provenance"] == ["ai-sdk-m8[jobcontext]"]
    assert extras["docling"] == []


def test_core_wheel_has_no_native_dependency_or_optional_import() -> None:
    """A clean interpreter can import the core package without extras
    installed, and it never imports ``cryptography`` or ``ai_sdk_m8``."""
    script = (
        "import sys, doctools_m8  # noqa: F401\n"
        "assert 'cryptography' not in sys.modules\n"
        "assert 'ai_sdk_m8' not in sys.modules\n"
        "assert 'pypdfium2' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True)
