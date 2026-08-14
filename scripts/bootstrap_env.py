#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from zs_scopesim_tools.cli import git_dirty, is_git_repo, load_manifest
from zs_scopesim_tools.paths import resolve_manifest_path, resolve_src_dir, repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update a ZShooter simulator environment.")
    parser.add_argument("--manifest", default="env/zshooter-stack.toml")
    parser.add_argument("--manager", choices=("mamba", "conda", "venv", "existing"), default="mamba")
    parser.add_argument("--env-name", default="zssim")
    parser.add_argument("--venv", default=".venv", help="venv path when --manager venv is used.")
    parser.add_argument("--src-dir", help="Source checkout directory. Defaults to manifest paths.src_dir.")
    parser.add_argument("--python", help="Python version. Defaults to manifest python.recommended.")
    parser.add_argument("--sync-refs", action="store_true",
                        help="Fetch and checkout manifest refs for existing managed repos.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Allow --sync-refs to checkout dirty existing repos.")
    parser.add_argument("--no-fetch", action="store_true", help="Skip git fetch when --sync-refs is used.")
    parser.add_argument("--no-clone-missing", action="store_true",
                        help="Do not clone managed repos that are not already present.")
    parser.add_argument("--recreate-env", action="store_true",
                        help="Remove and recreate the selected conda/mamba environment or venv.")
    args = parser.parse_args()

    manifest = load_manifest(resolve_manifest_path(args.manifest))
    src_dir = resolve_src_dir(manifest, args.src_dir)
    python_version = args.python or manifest.get("python", {}).get("recommended")
    env_python = prepare_environment(args.manager, args.env_name, Path(args.venv), python_version,
                                     recreate=args.recreate_env)

    failures = 0
    editable_paths = []
    src_dir.mkdir(parents=True, exist_ok=True)
    for name, repo in manifest.get("repos", {}).items():
        if not repo.get("managed", True):
            continue
        path = repo_path(repo, src_dir)
        created = ensure_repo(name, repo, path, clone_missing=not args.no_clone_missing)
        if not path.exists() or not is_git_repo(path):
            failures += 1
            continue
        if created:
            checkout_ref(name, path, repo["ref"], fetch=False, allow_dirty=True)
        elif args.sync_refs:
            result = checkout_ref(name, path, repo["ref"], fetch=not args.no_fetch, allow_dirty=args.allow_dirty)
            if result != 0:
                failures += 1
                continue
        else:
            print(f"{name}: leaving existing checkout unchanged at {path}")
        if repo.get("editable", True):
            editable_paths.append(path)

    if failures:
        print(f"Repository setup failed for {failures} managed checkout(s).", file=sys.stderr)
        return 1

    palace_path = REPO_ROOT / manifest.get("local", {}).get("palace", {}).get("path", "PALACE")
    editable_paths.extend([palace_path, REPO_ROOT])
    packages = manifest.get("pip", {}).get("packages", [])
    install_cmd = [str(env_python), "-m", "pip", "install"]
    for path in editable_paths:
        install_cmd.extend(["-e", str(path)])
    install_cmd.extend(packages)
    run(install_cmd)

    run([str(env_python), "-m", "ipykernel", "install", "--user", "--name", args.env_name,
         "--display-name", f"Python ({args.env_name})"])
    return 0


def prepare_environment(manager: str, env_name: str, venv_path: Path, python_version: str | None, *,
                        recreate: bool = False) -> Path:
    if manager == "existing":
        if recreate:
            raise ValueError("--recreate-env cannot be used with --manager existing")
        return Path(sys.executable)
    if manager in {"mamba", "conda"}:
        exists = conda_env_exists(manager, env_name)
        if recreate and exists:
            run([manager, "env", "remove", "-y", "-n", env_name])
            exists = False
        if exists:
            print(f"{manager}: using existing environment {env_name}")
        else:
            create_cmd = [manager, "create", "-y", "-n", env_name]
            if python_version:
                create_cmd.append(f"python={python_version}")
            create_cmd.append("pip")
            run(create_cmd)
        return Path(run([manager, "run", "-n", env_name, "python", "-c",
                         "import sys; print(sys.executable)"], capture=True).stdout.strip())
    venv = venv_path.expanduser().resolve()
    if recreate and venv.exists():
        if venv in {Path.home().resolve(), REPO_ROOT.resolve()} or not (venv / "pyvenv.cfg").is_file():
            raise ValueError(f"Refusing to remove non-venv path: {venv}")
        shutil.rmtree(venv)
    if not (venv / "bin" / "python").exists():
        cmd = [sys.executable, "-m", "venv", str(venv)]
        run(cmd)
    return venv / "bin" / "python"


def conda_env_exists(manager: str, env_name: str) -> bool:
    result = run([manager, "env", "list", "--json"], check=False, capture=True)
    if result.returncode == 0:
        try:
            envs = json.loads(result.stdout).get("envs", [])
        except json.JSONDecodeError:
            envs = []
        for env_path in envs:
            if Path(env_path).name == env_name:
                return True
    return run([manager, "run", "-n", env_name, "python", "-c", "pass"], check=False, capture=True).returncode == 0


def ensure_repo(name: str, repo: dict, path: Path, clone_missing: bool) -> bool:
    if not path.exists():
        if not clone_missing:
            print(f"{name}: missing at {path}; skipping because --no-clone-missing was supplied", file=sys.stderr)
            return False
        run(["git", "clone", repo["url"], str(path)])
        return True
    if not is_git_repo(path):
        print(f"{name}: invalid checkout at {path}; run zs-sim doctor --fix", file=sys.stderr)
        return False
    print(f"{name}: using {path}")
    return False


def checkout_ref(name: str, path: Path, ref: str, fetch: bool, allow_dirty: bool) -> int:
    if git_dirty(path) and not allow_dirty:
        print(f"{name}: dirty checkout at {path}; refusing to checkout {ref}", file=sys.stderr)
        print(f"{name}: rerun with --allow-dirty only if you intend to move this checkout", file=sys.stderr)
        return 1
    if fetch:
        result = run(["git", "-C", str(path), "fetch", "--tags", "origin"], check=False)
        if result.returncode != 0:
            print(f"{name}: fetch failed; attempting checkout using local refs", file=sys.stderr)
    run(["git", "-C", str(path), "checkout", ref])
    return 0


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    kwargs = {"text": True}
    if capture:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
