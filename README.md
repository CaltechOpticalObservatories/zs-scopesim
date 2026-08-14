# ZShooter ScopeSim Workspace

This repository contains all the user-facing simulator products for the ZShooter project, including 
tutorial and science-validation notebooks, simulator documentation, and project-specific tools/helpers used to run the workflow.
It also includes a local copy of [PALACE](https://ui.adsabs.harvard.edu/abs/2025GMD....18.4353N/abstract) for modeling airglow emission.

ZShooter is simulated with the ScopeSim framework, which observes simulated light sources through the atmosphere, telescope and instrument to produce raw detector images.
The raw data is processed with a Pyreduce-based barebones pipeline to extract quick-look 1D spectra.

The simulator setup uses Caltech Optical Observatories versions of [ScopeSim](https://github.com/CaltechOpticalObservatories/ScopeSim/) and [irdb](https://github.com/CaltechOpticalObservatories/irdb). 
To keep this repo synced with latest changes in ScopeSim and IRDB, these packages are installed as editable checkouts so users can inspect the code and developers can reproduce science-ready environment.

## Quick start

Clone the repository and set up an environment:

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
without fetching or checking out branches/tags. This helps avoid changing
developer's and science-user's working trees safe. To explicitly update managed checkouts to the
versions listed in `env/zshooter-stack.toml`, run:

```bash
scripts/sync_stack.sh --install
```

Check that the environment is set up correctly:

```bash
zs-sim doctor
```

Launch the notebooks:

```bash
scripts/run_notebooks.sh
```

## Keeping checkouts in sync

The recommended checkout refs are listed in
`env/zshooter-stack.toml`. To sync the COO ScopeSim and irdb forks to those
refs, use:

```bash
scripts/sync_stack.sh
```

If a checkout has local changes, it will not be updated unless you pass the `--allow-dirty` flag.

When reporting issues, include the output of:

```bash
zs-sim doctor
```

## Documentation

Developer and user documentation lives in `docs/` and is built with Sphinx.
The layout is kept intentionally small and structured so it can be published on the ZShooter project site later.
