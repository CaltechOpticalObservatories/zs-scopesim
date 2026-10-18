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
reproducibility
troubleshooting
```
