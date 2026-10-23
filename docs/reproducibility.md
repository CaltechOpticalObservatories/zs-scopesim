# Reproducibility

The file `env/zshooter-stack.toml` records expected refs for managed editable
repositories. This allows science users and developers to report and reproduce
the same simulator state.

The review-series manifest pins ScopeSim, IRDB, zshooter-drp and PyReduce to
commits published in the CaltechOpticalObservatories forks. The spectroscopy
stack contains the adopted upstream baseline and selected shared numerical
fixes. See [the X-shooter assessment](xshooter-validation.md) for the executed
evidence and scientific approximations.

The `PyReduce-xshooter` directory remains a separate checkout so an older,
independently modified `~/src/PyReduce` checkout is preserved. Its remote in
this manifest is the Caltech fork containing the covariance changes; the
upstream repository does not contain those feature commits. Restart existing
notebook kernels after syncing or changing editable checkouts.

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
