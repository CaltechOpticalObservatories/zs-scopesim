from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from zs_scopesim_tools.paths import repo_path, resolve_manifest_path, resolve_src_dir


DEFAULT_MANIFEST = "env/zshooter-stack.toml"
DEFAULT_IMPORTS = (
    ("numpy", "numpy"),
    ("astropy", "astropy"),
    ("matplotlib", "matplotlib"),
    ("scopesim", "ScopeSim"),
    ("scopesim_templates", "ScopeSim_Templates"),
    ("irdb", "irdb"),
    ("spextra", "speXtra"),
    ("skycalc_ipy", "skycalc_ipy"),
    ("pyckles", "Pyckles"),
    ("palace", "palace"),
)
SYNC_STATUS = {
    "no_ref": "no ref specified; skipping",
    "missing_url": "missing and no url is specified",
    "dirty": "dirty worktree at {path}; use --allow-dirty to override",
    "fetch": "fetching tags",
    "checkout": "checking out {ref}",
    "install": "installing editable",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zs-sim")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to stack manifest. Default: {DEFAULT_MANIFEST}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Report environment, package, and git state.")
    doctor_parser.add_argument("--src-dir", help="Override source checkout directory.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    doctor_parser.set_defaults(func=cmd_doctor)

    sync_parser = subparsers.add_parser("sync-tags", help="Checkout manifest refs for managed editable repos.")
    sync_parser.add_argument("--src-dir", help="Override source checkout directory.")
    sync_parser.add_argument("--no-fetch", action="store_true", help="Do not fetch tags before checkout.")
    sync_parser.add_argument("--allow-dirty", action="store_true", help="Checkout refs even if a repo is dirty.")
    sync_parser.add_argument("--install", action="store_true", help="Run editable pip installs after checkout.")
    sync_parser.set_defaults(func=cmd_sync_tags)

    launch_parser = subparsers.add_parser("launch-notebooks", help="Start Jupyter in the notebook directory.")
    launch_parser.add_argument("--notebook-dir", default="notebooks", help="Notebook directory to open.")
    launch_parser.add_argument("--port", type=int, default=8888)
    launch_parser.add_argument("--host", default="127.0.0.1")
    launch_parser.add_argument("--no-browser", action="store_true")
    launch_parser.add_argument("--app", choices=("lab", "notebook"), default="lab")
    launch_parser.set_defaults(func=cmd_launch_notebooks)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_doctor(args: argparse.Namespace) -> int:
    manifest, src_dir, manifest_path = load_cli_context(args)
    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "constraint": manifest.get("python", {}).get("constraint"),
            "recommended": manifest.get("python", {}).get("recommended"),
        },
        "manifest": str(manifest_path),
        "src_dir": str(src_dir),
        "packages": package_report(),
        "palace_resources": palace_resource_report(),
        "repos": repo_report(manifest, src_dir),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 0


def cmd_sync_tags(args: argparse.Namespace) -> int:
    manifest, src_dir, _ = load_cli_context(args)
    failures = 0
    for name, repo, path in iter_repos(manifest, src_dir, managed_only=True):
        url = repo.get("url")
        ref = repo.get("ref")
        if not isinstance(ref, str) or not ref:
            print(f"{name}: {SYNC_STATUS['no_ref']}")
            continue
        if not path.exists():
            if not isinstance(url, str) or not url:
                print(f"{name}: {path} {SYNC_STATUS['missing_url']}")
                failures += 1
                continue
            print(f"{name}: cloning {url} into {path}")
            run(["git", "clone", url, str(path)], check=True)
        dirty = git_dirty(path)
        if dirty and not args.allow_dirty:
            print(f"{name}: {SYNC_STATUS['dirty'].format(path=path)}")
            failures += 1
            continue
        if not args.no_fetch:
            print(f"{name}: {SYNC_STATUS['fetch']}")
            run(["git", "-C", str(path), "fetch", "--tags", "origin"], check=False)
        print(f"{name}: {SYNC_STATUS['checkout'].format(ref=ref)}")
        result = run(["git", "-C", str(path), "checkout", ref], check=False)
        if result.returncode != 0:
            failures += 1
            continue
        if args.install and repo.get("editable", True):
            print(f"{name}: {SYNC_STATUS['install']}")
            result = run([sys.executable, "-m", "pip", "install", "-e", str(path)], check=False)
            if result.returncode != 0:
                failures += 1
    return 1 if failures else 0


def cmd_launch_notebooks(args: argparse.Namespace) -> int:
    notebook_dir = Path(args.notebook_dir).expanduser()
    if not notebook_dir.exists():
        print(f"Notebook directory not found: {notebook_dir}", file=sys.stderr)
        return 1
    module = "jupyterlab" if args.app == "lab" else "notebook"
    cmd = [
        sys.executable,
        "-m",
        module,
        str(notebook_dir),
        "--ip",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.no_browser:
        cmd.append("--no-browser")
    return run(cmd, check=False).returncode


def load_cli_context(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = resolve_manifest_path(args.manifest)
    manifest = load_manifest(manifest_path)
    src_dir = resolve_src_dir(manifest, args.src_dir)
    return manifest, src_dir, manifest_path


def iter_repos(manifest: dict[str, Any], src_dir: Path, *, managed_only: bool = False
               ) -> Iterator[tuple[str, dict[str, Any], Path]]:
    for name, repo in manifest.get("repos", {}).items():
        if managed_only and not repo.get("managed", True):
            continue
        yield name, repo, repo_path(repo, src_dir)


####### Manifest and diagnostics helpers ########

def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("rb") as handle:
        return tomllib.load(handle)

def package_report() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for module_name, dist_name in DEFAULT_IMPORTS:
        entry: dict[str, Any] = {"importable": False}
        try:
            module = importlib.import_module(module_name)
            entry["importable"] = True
            entry["file"] = getattr(module, "__file__", None)
            entry["version"] = getattr(module, "__version__", None)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail on broken imports
            entry["error"] = f"{type(exc).__name__}: {exc}"
        try:
            entry["distribution_version"] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            pass
        report[module_name] = entry
    return report


def palace_resource_report() -> dict[str, Any]:
    resources = {
        "palace.config": ["palace_default.par"],
        "palace.data": ["palace_lines.fits", "palace_cont.fits", "palace_var.fits"],
    }
    report: dict[str, Any] = {}
    for package, names in resources.items():
        package_report_entry: dict[str, Any] = {}
        try:
            root = importlib.resources.files(package)
            for name in names:
                package_report_entry[name] = root.joinpath(name).is_file()
        except (ImportError, FileNotFoundError, AttributeError, TypeError) as exc:
            package_report_entry["error"] = f"{type(exc).__name__}: {exc}"
        report[package] = package_report_entry
    return report


def repo_report(manifest: dict[str, Any], src_dir: Path) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, repo, path in iter_repos(manifest, src_dir):
        entry: dict[str, Any] = {
            "path": str(path),
            "expected_ref": repo.get("ref"),
            "exists": path.exists(),
        }
        if path.exists():
            entry["head"] = git_output(path, ["rev-parse", "HEAD"])
            entry["branch"] = git_output(path, ["branch", "--show-current"])
            entry["tags"] = git_output(path, ["tag", "--points-at", "HEAD"]).splitlines()
            entry["dirty"] = git_dirty(path)
        report[name] = entry
    return report


def git_output(path: Path, args: list[str]) -> str:
    result = run(["git", "-C", str(path), *args], check=False, capture=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_dirty(path: Path) -> bool:
    result = run(["git", "-C", str(path), "status", "--porcelain"], check=False, capture=True)
    return bool(result.stdout.strip())


def run(cmd: list[str], check: bool, capture: bool = False) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {"text": True}
    if capture:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})
    result = cast(subprocess.CompletedProcess[str], subprocess.run(cmd, **kwargs))
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def print_text_report(report: dict[str, Any]) -> None:
    print("ZShooter simulator environment")
    print(f"Python: {report['python']['version']}")
    print(f"Executable: {report['python']['executable']}")
    print(f"Manifest: {report['manifest']}")
    print(f"Source dir: {report['src_dir']}")
    print()
    print("Packages")
    for name, entry in report["packages"].items():
        if entry["importable"]:
            version = entry.get("version") or entry.get("distribution_version") or "unknown"
            print(f"  {name}: ok ({version})")
        else:
            print(f"  {name}: missing or broken ({entry.get('error', 'unknown error')})")
    print()
    print("PALACE resources")
    for package, entry in report["palace_resources"].items():
        print(f"  {package}: {entry}")
    print()
    print("Repos")
    for name, entry in report["repos"].items():
        if not entry["exists"]:
            print(f"  {name}: missing at {entry['path']} expected={entry.get('expected_ref')}")
            continue
        tag_text = ",".join(entry.get("tags", [])) or "no tag"
        dirty = " dirty" if entry.get("dirty") else ""
        print(f"  {name}: {entry.get('branch') or 'detached'} {entry.get('head')} [{tag_text}]{dirty}")


if __name__ == "__main__":
    raise SystemExit(main())
