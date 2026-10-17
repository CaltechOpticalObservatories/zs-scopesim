# Science Notebooks

Science notebooks are in `notebooks/`. Development notebooks live in `dev_notebooks/`.

`01_Kilonovae_XSHOOTER.ipynb` is a detector-only companion for the three-arm X-shooter model. See [the configuration and accuracy notes](xshooter.md) before using it for quantitative comparisons.

Launch a notebook server with:

```bash
scripts/run_notebooks.sh
```

The script uses the `ZS_SIM_ENV` environment variable when set. Otherwise it
defaults to `zssim` for mamba-based workflows.
