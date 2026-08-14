from typing import Any
from pathlib import Path
import importlib.resources

def repo_root(start: str | Path | None = None) -> Path:
    """Return repository root given start path or using this file's parent."""
    path = Path(start).expanduser().resolve() if start else Path(__file__).resolve()
    candidates = [path] if path.is_dir() else [path.parent]
    candidates.extend(candidates[0].parents)
    for parent in candidates:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path.cwd().resolve()

def resolve_manifest_path(path: str) -> Path:
    """Return the absolute path to a manifest file, resolving relative to cwd or repo root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = repo_root() / candidate
    return repo_candidate

def resolve_src_dir(manifest: dict[str, Any], override: str | None) -> Path:
    """Return the absolute path to the source directory, using override if given, else manifest."""
    value = override or manifest.get("paths", {}).get("src_dir", "~/src")
    return Path(value).expanduser().resolve()

def repo_path(repo: dict[str, Any], src_dir: Path) -> Path:
    """Return the absolute path to a repo given its manifest entry and the src_dir."""
    raw_path = Path(repo["path"]).expanduser()
    if raw_path.is_absolute():
        return raw_path
    base = repo.get("relative_to", "src_dir")
    if base == "repo":
        return (repo_root() / raw_path).resolve()
    return (src_dir / raw_path).resolve()

def resolve_irdb_path(fallback_irdb_path: str | Path | None = None) -> Path:
    """Return the local IRDB checkout or installed editable package root."""
    try:
        resolved = Path(importlib.resources.files("irdb")).parent.resolve()
    except ModuleNotFoundError:
        if fallback_irdb_path is None:
            fallback_irdb_path = Path.home() / "src" / "irdb"
        resolved = Path(fallback_irdb_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved

def make_output_dir(dirname: str = "outputs",
                    base_path: str | Path | None = None,
                    avoid_path: str | Path | None = None) -> Path:
    """Return a repo-local output directory, avoiding accidental writes into certain paths if given."""
    cwd = Path(base_path).expanduser().resolve() if base_path else Path.cwd().resolve()
    if avoid_path is not None:
        avoid_path = Path(avoid_path).expanduser().resolve()
        if cwd == avoid_path or avoid_path in cwd.parents:
            cwd = repo_root()
    output_dir = cwd / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir