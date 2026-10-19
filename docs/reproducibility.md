# Reproducibility

The file `env/zshooter-stack.toml` records expected refs for managed editable
repositories. This allows science users and developers to report and reproduce
the same simulator state.

On the `xshooter` branch, ScopeSim and IRDB are pinned to the adopted upstream
merges and shared numerical fixes. The historical `fidelity-candidates` branches
remain as review records; their selected corrections are now in the feature stack.
See [the X-shooter assessment](xshooter-validation.md) for revisions,
validation results and remaining scientific approximations. The commits are local
until published; another machine needs those commits before it can sync them.

The kilonova reductions also pin the local DRP and `PyReduce-xshooter` revisions.
The latter is a separate checkout based on PyReduce 0.9, installed editable into
`zssim`; it preserves an older independently modified `~/src/PyReduce` checkout.
These local feature commits have not been pushed to the listed upstream remote.
Obtain the commits from this workspace before attempting to reproduce the
variance-aware reductions on another machine. Restart existing notebook kernels
after switching the installed PyReduce checkout.

The ScopeSim pin includes the pre-slit ADC correction `8727eafe` and its
zero-shift optimization `aef3a53d`. Restart kernels
to load it. The archived kilonova frames use the preceding `a0495f7` revision;
the [reduction assessment](kilonova-mirrored-reduction.md) separates those saved
results from the corrected ADC experiment and gives the full regeneration steps.

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
