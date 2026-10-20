# ZShooter ScopeSim

This documentation covers the ZShooter simulator workspace: environment setup,
science notebooks, reproducibility tools, and common troubleshooting.

For the current instrument comparison, start with the
[flat-AB comparison notebook](../notebooks/02_Flat_AB_Comparison.ipynb) and the
[executed results report](xshooter-comparison-results.md). The
[shared simulator assessment](xshooter-validation.md) links
the adopted fixes, tests, runtime measurements and remaining scientific approximations.
The [performance reconciliation](performance-reconciliation.md) compares committed
main, the preserved user rerun and the corrected execution. The
[slit-illumination lamp notebook](../notebooks/03_Slit_Illumination_Lamps.ipynb)
tests centered and quarter-slit-offset sources under natural seeing and AO.

The [kilonova reduction repair](../../zshooter-drp/docs/source/kilonova-reduction-repair.md)
records the updated DRP calibrations, executed extraction checks, thermal-background
correction to the standard, and the host-light explanation of the apparent BLUE flux excess.
The [mirrored kilonova reductions](kilonova-mirrored-reduction.md) describe the
shared electron-variance inputs, native optimal extraction and propagated spectral
covariance in the ZShooter and X-shooter examples.

The [resolution and sampling note](resolution-and-sampling.md) separates slit geometry,
point-source illumination, diffraction and native-pixel width measurements. The
[detector operating-point assessment](detector-operating-points.md) gives the
`SPEC`/`SPEC_ALTDETECT` packages, explicit per-arm exposure settings, charge-limit
warnings, avalanche variance and the specifications needed for vendor discussions.

```{toctree}
:maxdepth: 2
:caption: Contents

install
notebooks
zshooter-xdisp-geometry
xshooter
xshooter-comparison-results
xshooter-validation
performance-reconciliation
resolution-and-sampling
detector-operating-points
kilonova-mirrored-reduction
reproducibility
troubleshooting
```
