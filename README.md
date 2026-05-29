# ZShooter ScopeSim Workspace

This repository is the authoritative home for ZShooter ScopeSim science
notebooks, simulator documentation, and project-owned simulator workflow tools.
It also carries a vendored copy of PALACE for local airglow-model development.

The main simulator stack uses the Caltech Optical Observatories forks of
ScopeSim and irdb. Those packages are installed as editable checkouts so users
can inspect the code and developers can reproduce science-team environments.

## Quickstart

Clone this repository and create or update an environment:

```bash
git clone https://github.com/CaltechOpticalObservatories/zs-scopesim.git
cd zs-scopesim
python scripts/bootstrap_env.py --manager mamba --env-name zssim
```

For a local venv instead:

```bash
python scripts/bootstrap_env.py --manager venv --venv .venv
```

By default, bootstrap uses existing ScopeSim and irdb checkouts in `~/src`
without fetching or checking out branches/tags. This keeps developer and
science-user working trees safe. To explicitly move managed checkouts to the
refs in `env/zshooter-stack.toml`, run:

```bash
scripts/sync_stack.sh --install
```

Check the environment:

```bash
zs-sim doctor
```

Start notebooks:

```bash
scripts/run_notebooks.sh
```

## Reproducibility

Known-good editable checkout refs are recorded in
`env/zshooter-stack.toml`. To sync the COO ScopeSim and irdb forks to those
refs, use:

```bash
scripts/sync_stack.sh
```

Dirty checkouts are not moved unless `--allow-dirty` is supplied.

When reporting issues, include:

```bash
zs-sim doctor
```

## Documentation

Developer and user documentation lives in `docs/` and is built with Sphinx.
The layout is intentionally small and structured so the ZShooter project site
can stage these pages later.
