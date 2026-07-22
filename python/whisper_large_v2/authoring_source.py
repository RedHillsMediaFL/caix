"""Authenticated, archive-backed CoreAI Models authoring source."""

from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator


@dataclass(frozen=True)
class AuthoringSourceContract:
    schema: str
    repository: str
    revision: str
    python_root: str
    package_subtree: str
    package_tree: str


@dataclass(frozen=True)
class AuthenticatedAuthoringSource:
    contract: AuthoringSourceContract
    extraction_root: Path
    python_root: Path


class AuthoringSourceError(RuntimeError):
    """The CoreAI Models source does not match its pinned Git identity."""


_APPROVED_CONTRACT = AuthoringSourceContract(
    schema="caix.coreai-models-authoring-source.v1",
    repository="https://github.com/kylejfrost/coreai-models.git",
    revision="e666cdc9848fd17f41e43504bc574c8964812c9e",
    python_root="python/src",
    package_subtree="python/src/coreai_models",
    package_tree="b2803957eee13084d06924cfc567a770379234ae",
)
_CONTRACT_FIELDS = frozenset(AuthoringSourceContract.__dataclass_fields__)


def load_authoring_source_contract(path: Path) -> AuthoringSourceContract:
    """Load the tracked lock and reject any source-identity drift."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthoringSourceError("authoring source lock is missing or malformed") from error
    if not isinstance(payload, dict) or set(payload) != _CONTRACT_FIELDS:
        raise AuthoringSourceError("authoring source lock fields differ")
    try:
        contract = AuthoringSourceContract(
            **{field: payload[field] for field in _CONTRACT_FIELDS}
        )
    except TypeError as error:
        raise AuthoringSourceError("authoring source lock values are malformed") from error
    for field in _CONTRACT_FIELDS:
        actual = getattr(contract, field)
        expected = getattr(_APPROVED_CONTRACT, field)
        if actual != expected:
            raise AuthoringSourceError(f"authoring source drift at {field}")
    return contract


def _run_git(repository: Path, *arguments: str, label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuthoringSourceError(f"authoring source {label} could not be verified") from error
    return completed.stdout.strip()


def _verify_repository(
    repository: Path,
    contract: AuthoringSourceContract,
) -> None:
    if not repository.is_dir():
        raise AuthoringSourceError("authoring source repository is not a directory")
    top_level = Path(
        _run_git(repository, "rev-parse", "--show-toplevel", label="repository")
    ).resolve()
    if top_level != repository.resolve():
        raise AuthoringSourceError("authoring source repository root differs")

    remotes = _run_git(repository, "remote", label="repository remotes").splitlines()
    remote_urls: set[str] = set()
    for remote in remotes:
        urls = _run_git(
            repository,
            "remote",
            "get-url",
            "--all",
            remote,
            label="repository remotes",
        )
        remote_urls.update(urls.splitlines())
    if contract.repository not in remote_urls:
        raise AuthoringSourceError("authoring source repository identity differs")

    revision = _run_git(
        repository,
        "rev-parse",
        "--verify",
        f"{contract.revision}^{{commit}}",
        label="revision",
    )
    if revision != contract.revision:
        raise AuthoringSourceError("authoring source revision differs")
    tree = _run_git(
        repository,
        "rev-parse",
        f"{contract.revision}:{contract.package_subtree}",
        label="package_tree",
    )
    if tree != contract.package_tree:
        raise AuthoringSourceError("authoring source package_tree differs")
    tree_type = _run_git(
        repository,
        "cat-file",
        "-t",
        tree,
        label="package_tree",
    )
    if tree_type != "tree":
        raise AuthoringSourceError("authoring source package_tree is not a Git tree")


def _coreai_modules() -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if module is not None
        and (name == "coreai_models" or name.startswith("coreai_models."))
    }


def verify_coreai_models_imports(python_root: Path) -> int:
    """Require every loaded CoreAI Models module to resolve below one archive root."""
    package_root = (python_root / "coreai_models").resolve(strict=True)
    imported = _coreai_modules()
    if not imported:
        raise AuthoringSourceError("coreai_models was not imported from authenticated root")
    for name, module in imported.items():
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise AuthoringSourceError(
                f"coreai_models import outside authenticated root: {name}"
            )
        try:
            Path(origin).resolve(strict=True).relative_to(package_root)
        except (OSError, ValueError) as error:
            raise AuthoringSourceError(
                f"coreai_models import outside authenticated root: {name}"
            ) from error
    return len(imported)


def _archive_revision(
    repository: Path,
    contract: AuthoringSourceContract,
    destination: Path,
) -> None:
    archive = destination / "coreai-models-source.tar"
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                f"--output={archive}",
                contract.revision,
                contract.python_root,
            ],
            check=True,
            capture_output=True,
        )
        with tarfile.open(archive, mode="r:") as source:
            source.extractall(destination, filter="data")
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise AuthoringSourceError("authoring source revision archive failed") from error


@contextmanager
def authenticated_authoring_source(
    repository: Path,
    contract: AuthoringSourceContract,
    *,
    temp_root: Path,
) -> Iterator[AuthenticatedAuthoringSource]:
    """Import CoreAI Models only from an archive of the pinned Git commit."""
    if _coreai_modules():
        raise AuthoringSourceError("coreai_models import outside authenticated root")
    if not temp_root.is_dir():
        raise AuthoringSourceError("authoring source temp root is not a directory")
    _verify_repository(repository, contract)

    with tempfile.TemporaryDirectory(
        prefix="coreai-models-authenticated-",
        dir=temp_root,
    ) as temporary:
        extraction_root = Path(temporary)
        _archive_revision(repository, contract, extraction_root)
        python_root = extraction_root / contract.python_root
        package_root = extraction_root / contract.package_subtree
        try:
            package_stat = os.lstat(package_root)
        except OSError as error:
            raise AuthoringSourceError("archived coreai_models package is missing") from error
        if not stat.S_ISDIR(package_stat.st_mode):
            raise AuthoringSourceError("archived coreai_models package is not a directory")

        source = AuthenticatedAuthoringSource(
            contract=contract,
            extraction_root=extraction_root,
            python_root=python_root,
        )
        sys.path.insert(0, str(python_root))
        importlib.invalidate_caches()
        try:
            importlib.import_module("coreai_models")
            verify_coreai_models_imports(python_root)
            yield source
            verify_coreai_models_imports(python_root)
        finally:
            sys.path.remove(str(python_root))
            for name in tuple(_coreai_modules()):
                sys.modules.pop(name, None)
            importlib.invalidate_caches()
