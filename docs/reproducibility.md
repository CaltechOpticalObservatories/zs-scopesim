# Reproducibility

The file `env/zshooter-stack.toml` records expected refs for managed editable
repositories. This allows science users and developers to report and reproduce
the same simulator state.

Sync managed checkouts:

```bash
scripts/sync_stack.sh
```

Install after syncing:

```bash
scripts/sync_stack.sh --install
```

The sync command refuses to move dirty checkouts unless `--allow-dirty` is
provided.

Report current state:

```bash
zs-sim doctor
```

For issue reports, include the doctor output and the notebook that triggered
the problem.
