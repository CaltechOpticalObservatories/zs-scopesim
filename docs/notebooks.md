# Science Notebooks

Science notebooks are in `notebooks/`. Development notebooks live in `dev_notebooks/`.

`01_Kilonovae_XSHOOTER.ipynb` is a detector-only companion for the three-arm X-shooter model. See [the configuration and accuracy notes](xshooter.md) before using it for quantitative comparisons.

`02_Flat_AB_Comparison.ipynb` is the technical comparison: a flat AB continuum,
common atmosphere, native boxcar extraction, measured line resolution,
covariance-aware degradation to approximately R=1000, limiting magnitude and
extracted throughput. OH wavelength selection is a separate diagnostic.
The [executed results](xshooter-comparison-results.md) summarize the plots and
measurements. The [current assessment](xshooter-validation.md) explains the
shared-engine evidence and limitations.

Launch a notebook server with:

```bash
scripts/run_notebooks.sh
```

The script uses the `ZS_SIM_ENV` environment variable when set. Otherwise it
defaults to `zssim` for mamba-based workflows.
