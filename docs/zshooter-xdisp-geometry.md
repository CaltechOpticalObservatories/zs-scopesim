# ZShooter cross-disperser geometry correction

**The automatic rotation tuning below has been withdrawn.** The original detector angles are restored in IRDB, including on the independent feature branch. The trial's coverage-only objective accepted excessive tilts and lost orders; it was insufficient for selecting a satisfactory echellogram layout. The third row of the figure and the `retuned` CSV rows record that rejected trial, not the active prescription.

The fixed-groove angle estimator did not satisfy the camera projection used by the simulator. Its correction and the restoration of the YJ grating to 350 grooves/mm remain separate commits. The rotation objective and detector-placement calculation are under review before further tuning.

The two planar dispersion calculations and empirical detector rotations remain intentional approximations. The existing 225 mm camera, 270 mm effective echelle dispersion focal length and configured plate scale already represent the adopted white-pupil prescription. Jason's revised H orders are not included: their values still need to come from the updated Zemax prescription.

## The calculation

The configured camera uses $y=f\tan(\beta-\beta_c)$. Putting the wavelength endpoints at $y=\pm L/2$ therefore requires

$$
h=\arctan\frac{L}{2f},\qquad
\cos\beta_c=\frac{\lambda_{\max}-\lambda_{\min}}{2d\sin h},\qquad
\alpha=\arcsin\left(\frac{\lambda_{\min}}d-\sin(\beta_c-h)\right).
$$

Here $d$ is groove spacing. This follows by subtracting the two planar grating equations. Negative incidence angles are valid signed solutions for some channels.

The old estimator compressed the requested cross-dispersion span by approximately 6.3%, 5.9%, 13.3%, 4.9%, 11.1% and 4.6% in B/G/R/YJ/H/K with the former prescription. The YJ change from 350 to 325 grooves/mm helped the old layout fit by exploiting this unintended dependence of span on ruling. The corrected estimator places both endpoints on the specified unrotated detector boundaries.

## Rejected detector-placement trial

The rejected search maximized the union of wavelengths within the configured band for which the complete slit fits in at least one order. That objective allowed steep YJ/K tilts and the complete loss of G/R traces. Coverage was measured by intersecting continuous segments from the actual generated traces with the detector rectangle. Overlapping orders counted a wavelength once. The search covered -30 to +30 degrees and refined the best neighborhood to 0.01 degree. Numerical maximization of that score was not an adequate acceptance test for the intended geometry adjustment.

| Band | Restored angle | Rejected angle | Baseline full-slit coverage | Corrected, original angles | Rejected full-slit coverage | Rejected slit-center coverage |
|---|---:|---:|---:|---:|---:|---:|
| B | -6.00 | -6.86 | 100.00% | 99.62% | 100.00% | 100.00% |
| G | -6.00 | -5.46 | 100.00% | 99.17% | 99.79% | 100.00% |
| R | -2.00 | -3.35 | 99.78% | 97.57% | 98.52% | 98.54% |
| YJ | -4.00 | +8.56 | 100.00% | 95.60% | 97.43% | 98.01% |
| H | -3.00 | -9.48 | 100.00% | 92.18% | 95.15% | 97.61% |
| K | -7.00 | +4.49 | 89.01% | 80.46% | 86.08% | 87.63% |

![Identical-scale detector footprints before and after correction](../validation/zshooter_xdisp/data/detector_footprints.png)

Black rectangles are detector boundaries; gray curves are the slit ends; colored curves are slit centers, with wavelength increasing from purple to yellow within each channel. The configured orders remain unchanged. Two traces, `g_85` and `r_56`, cease to intersect the detector after retuning; the trace generator consequently omits them.

The retained losses are concrete: G loses full-slit coverage above 649.53 nm, R below 654.38 nm plus a small internal gap, and YJ above 1330.24 nm. H has full-slit gaps at 1470-1471.46, 1488.36-1500.34 and 1825.99-1830 nm. K already had substantial gaps between orders in the baseline; retuning trades some center coverage for improved full-slit coverage. Every interval is listed in [gaps.csv](../validation/zshooter_xdisp/data/gaps.csv).

The minimum vertical separation of neighboring geometric slit edges was positive in every trial channel. YJ was tightest at 0.29 pixel, compared with a 3.92-pixel geometric overlap in the baseline. These measurements do not resolve the rejected layout's excessive tilt and missing-order problems. Detector placement must be checked before interpreting the trial's coverage as an instrument prediction.

## Sampling and verification

The unrotated echelle wavelength-to-x mapping agrees exactly with the baseline at three detector positions in every configured order of all six channels. At matched trace wavelengths, the change in along-trace wavelength sampling is below 0.6%. Detector-x sampling changes by up to 3.0% in YJ and 2.5% in K because the traces are more tilted. These coordinate-specific sampling measurements are in [sampling.csv](../validation/zshooter_xdisp/data/sampling.csv); they are not measurements of a simulated line FWHM.

The ScopeSim checks pass **22 tests**, covering all six endpoint projections, inverse mappings, units, negative incidence, infeasible angles, explicit-angle and free-frequency paths, and trace generation. An existing test that asserted identical wavelength limits after rotation also failed on the baseline; it now tests the actual detector-clipping behavior. The coverage calculation passes **6 tests**, including analytically known rectangle intersections, sampling independence, overlapping orders, signed slit separation and missing configuration.

The offline rotation search took 74.6 seconds. Normal observations read the saved angles and perform no search. Regenerating the comparisons with saved rotations takes about 7.6 seconds including plots and FITS tables.

## Independent commits and reproduction

All three repositories use branch `fix/zshooter-xdisp-geometry`. ScopeSim starts at `main` (`4830d68a`), IRDB at `dev_master` (`d1415c4`), and this workspace at `main` (`6a616ea`). The feature commits are:

- ScopeSim `acfa5a71`: estimator correction and tests.
- IRDB `cf88595`: restored YJ ruling.
- IRDB `aefd527`: rejected detector rotation trial, reverted by `cf9110a`.

Use the corrected ScopeSim and IRDB branches together. The executed assessment imported ScopeSim from `/private/tmp/zshooter-xdisp/scopesim/scopesim/__init__.py` with the configured `zssim` interpreter. The [assessment code](../validation/zshooter_xdisp/geometry.py), [coverage table](../validation/zshooter_xdisp/data/coverage.csv) and [rejected trial angles](../validation/zshooter_xdisp/data/rotations.csv) are retained to inspect the failed trial. **Do not copy those trial angles into an instrument configuration.**

To reproduce the rejected trial for investigation, use the following commands. Set `PYTHONPATH` to the corrected ScopeSim checkout if it is not the installed editable checkout.

```sh
git -C ../ScopeSim show 4830d68a:scopesim/optics/echelle.py > /tmp/zshooter-baseline-echelle.py
git -C ../irdb show d1415c4:ZShooter_v2/traces/echelle_trace_parameters.dat > /tmp/zshooter-baseline-parameters.dat
conda run -n zssim python validation/zshooter_xdisp/geometry.py \
  --parameters ../irdb/ZShooter_v2/traces/echelle_trace_parameters.dat \
  --baseline-parameters /tmp/zshooter-baseline-parameters.dat \
  --baseline-echelle /tmp/zshooter-baseline-echelle.py \
  --rotations validation/zshooter_xdisp/data/rotations.csv \
  --output validation/zshooter_xdisp/data
```

Omit `--rotations` to repeat the search. Generated trace FITS tables are local artifacts; the report, plots and numerical summaries are versioned.
