# Installation

The recommended workflow keeps the COO forks of ScopeSim and irdb as editable
checkouts. The stack manifest at `env/zshooter-stack.toml` records the refs
used by this repository.

## Mamba Environment

```bash
python scripts/bootstrap_env.py --manager mamba --env-name zssim
```

If the environment already exists, the bootstrap script reuses it.

## Venv Environment

```bash
python scripts/bootstrap_env.py --manager venv --venv .venv
```

## Checkout Safety

Bootstrap_env installs from existing editable checkouts but does not fetch or
checkout refs for those repos by default. This protects local work in
`~/src/ScopeSim` and `~/src/irdb`.

To make bootstrap also move existing checkouts to the manifest refs, opt in
explicitly:

```bash
python scripts/bootstrap_env.py --manager mamba --env-name zssim --sync-refs
```

Dirty checkouts are refused unless `--allow-dirty` is also supplied.

## Validation

```bash
zs-sim doctor
```

The doctor command prints Python version, import state, PALACE resource checks,
and git refs for managed editable checkouts.
