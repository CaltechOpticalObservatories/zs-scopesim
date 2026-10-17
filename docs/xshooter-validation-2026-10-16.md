# X-shooter comparison and shared simulator fidelity

**Historical assessment.** See [the current assessment](xshooter-validation.md) for the merged upstream stack, adopted shared fixes, comparison notebook and current tests. Candidate status and counts below describe the earlier revisions only.

Assessment date: 2026-10-16. The feature implementation produces native X-shooter detector frames with configurable CCD binning. It is useful for controlled observation forecasts, but **it does not yet validate the absolute fidelity of ZShooter simulations**. Shared source rendering, aperture sampling, and spectral sampling still have measurable errors. The candidate corrections remain separate from the notebook's pinned feature stack.

## What transfers between instruments

Measured instrument response belongs in IRDB. An empirical X-shooter response is appropriate even when no coating model is available; it need not be degraded to resemble an unfinished ZShooter design. However, a response inferred from a standard exposure can absorb finite-aperture losses, transparency errors, and numerical errors in the calculation used to infer it. Reproducing that same exposure is calibration, not an independent test of the engine.

The feature commits separate reusable calculations from instrument inputs:

| Calculation | X-shooter input or use | Implication for ZShooter |
|---|---|---|
| Independent detector gains, readout modes and headers | Three arm configurations | Generic routing, usable by any instrument |
| Charge binning, detector reset and focal-plane WCS | Documented CCD on-chip modes | Does not establish that ZShooter hardware supports charge binning |
| Read-noise interpolation | Measured NIR noise versus DIT | Requires an appropriate detector measurement for another instrument |
| Slit dispersion and optical convolution | ESO physical-model derivative; optical blur fitted to the manual | Generic mapper inputs; do not infer a ZShooter blur from a nominal resolving power |
| Registered detector background | Empirical NIR warm-background approximation | Calibration specific to X-shooter, not validation of shared thermal physics |
| Cached SkyCalc units | FITS wavelength units, or an explicit unit for legacy caches | Common unit correctness; ZShooter's separate TAPAS transmission is unchanged |
| Plane-by-plane atmospheric interpolation | Same spatial interpolation without spectral mixing or large coordinate arrays | Performance/correctness of that interpolation path; does not fix integer-roll ADC residuals |

The optical response and resolving-power table are fitted inputs. The second standard-star nod tests repeatability within the same pair. Independent trace samples test interpolation against the ESO optical model. Analytic photon, noise, and line-width tests check specified numerical relationships. These are different kinds of evidence and should be reported separately.

For an observation comparison, hold the source SED and normalization, centering, exposure convention, and observing assumptions explicit. Keep the native sampling, slit choices, detector noise, and site/telescope differences visible. Total observing time, DIT, NDIT, and on-chip versus software binning are not interchangeable. A comparison isolating only the spectrographs is a different experiment from comparing each instrument at its actual telescope and site. No common-resolution extraction or reduction pipeline is supplied here.

## Revisions and commit boundaries

The original working trees were saved before splitting the changes. All three main checkouts remain on `xshooter`; unrelated science notebook directories were left alone. Nothing was pushed.

| Repository | Original base | Feature tip selected by the notebook |
|---|---|---|
| zs-scopesim | `6a616ea` | This document and companion notebook are committed on `xshooter` |
| ScopeSim | `4830d68a` | `c0e0dfdc42df8972be4f4458abdf7d757bd54273` |
| IRDB | `d1415c4` | `b966cdda999d2891967227894f0cab7473d031d7` |

ScopeSim's eight feature commits cover detector-arm configuration; charge binning/reset/WCS; noise and exposure metadata; calibrated spectral traces; registered backgrounds; cached-sky units; atmospheric interpolation; and explicit legacy-cache units. IRDB's four commits cover Paranal/UT3; attributed calibration inputs and preparation; three-arm optics; and full-frame readout/noise/tests. The source-model download correction is a separate zs-scopesim commit.

Candidate tips, deliberately not selected in `env/zshooter-stack.toml`:

- ScopeSim `fidelity-candidates`: `6addf0973f888bf1297c41579d69348361cde276`.
- IRDB `fidelity-candidates`: `fde20be51fcd2680f34484c3b9c59c514bec299b`.

The candidate worktrees used for this review are `/tmp/xshooter-review/candidate-scopesim` and `/tmp/xshooter-review/candidate-irdb`. The branch refs remain in the original repositories; these temporary working directories are not dependencies of the science notebook.

These branches retain the same X-shooter throughput calibration. They add WCS-derived slit coordinates, explicit local IFU-slice centers, an opt-in physical slit derivative from ZShooter's existing analytical spectrograph, detector-area units in the photon Jacobian, corrected software-binning axes/WCS, and Poisson sampling of ZShooter dark current. The ZShooter prescription opts into physical slit dispersion and the existing trace Jacobian. The derivative assumes equal along- and across-slit magnification; it is not a complete optical prescription. No ZShooter optical blur or on-chip readout was invented.

The software-binning correction retains its position after read noise: summing two pixels with independent 3-electron read noise gives variance 18, whereas one CCD amplifier read after on-chip binning gives variance 9. The tests preserve this physical distinction.

## Review of the supplied flux report

The Fremling/Claude report describes older revisions (`5f7316b6`, `0c40ab2`) and references local fixes and 331 tests that were not available here. Those claims were not accepted as evidence of the current branches. The supplied witnesses were rerun against the feature and candidate checkouts with only a fixed observation date, MJD 60310.25, added. The same TAPAS data were used. [Measurements](../validation/xshooter/data/report_witnesses.txt) and [exit codes](../validation/xshooter/data/witness_status.csv) are saved; exit 1 means the witness found its reported discrepancy.

| Report item | Result on the current code and disposition |
|---|---|
| ZS-01, repeated atmospheric transmission | Already fixed before this work. Setting transmission to 0.5 changes counts by 0.5, not 0.25. Both branches pass. |
| ZS-02, rasterized slit width | Reproduced. A configured 0.70-arcsec slit spans 0.6282 arcsec on the nominal grid; tested width errors range from -15.1% to +12.2%. Requiring an odd row count is not itself a physical correctness criterion. No aperture-rendering overhaul was adopted. |
| ZS-03, point-source bilinear broadening | Code-reviewed; the added phase-dependent variance is real. The proposed replacement renderer was not implemented or independently rerun here. |
| ZS-04, point-sampled Moffat | Reproduced. In-slit/exact ratios are 1.8147 and 1.2017 for 0.10- and 0.14-arcsec AO FWHM. The ordinary-seeing case is also affected; see below. |
| ZS-05, narrow-profile detector quadrature | Pixel-center integration remains an approximation. The report's exact renderer and its claimed tests were not available. No blanket adoption. |
| ZS-06, airmass-independent TAPAS | Already fixed before this work. Both branches reproduce the tested relation between transmissions at airmasses one and two. |
| ZS-07, point-sampled spectra | Reproduced for unresolved lines; strongly dependent on line width relative to the configured grid. Current spacing is 0.1 Angstrom, not the older report's 0.05. |
| ZS-09, image projection | The current uniform-image witness is correct in every row, but its signed-position witness is mirrored. The report's broader image/cube claims were not all independently exercised. |
| ZS-10, off-slit illumination | Reproduced: two sources outside a 0.33-arcsec slit contribute zero instead of analytic fractions 0.2801 and 0.1668. Material for host galaxies, neighbours, and pointing errors. |
| ZS-11, slit coordinates | Reproduced: a source at 2 arcsec maps to 1.9730 arcsec. Candidate maps it to 2.0000. IFU regression testing required subtracting each real aperture's center rather than treating sky position as local cross-slit displacement. |
| ZS-12, slit ends | Endpoint interpolation remains a concern; this witness was not rerun. |
| ZS-13/14, point placement and edges | Reproduced: integer placement chooses the wrong column and an in-slit edge source raises `IndexError` with subpixel splitting. No new renderer was substituted. |
| ZS-15, integer ADC roll | Reproduced: a -0.0117-arcsec residual rounds to zero, while the analytic slit flux changes. The plane-interpolation optimization does not fix this roll path. |
| ZS-16, sky grid versus detector scale | Candidate improves the full-scene half-grid/nominal flux ratio from 0.54746 to 1.09501, but does not make it unity. See the adoption limit below. |
| ZS-17/18, wavelength averaging and PSF truncation | Known approximations; not independently quantified in this pass. |
| ZS-19, dispersion derivative | A gradient norm alone does not supply an optical cross-slit derivative. Candidate uses the configured unrotated analytical spectrograph with an explicit magnification assumption. |
| ZS-20/21, discrete PSFs | Broad changes were deferred. Real MICADO, METIS, MOSAIC and HAWKI observations test the limited port actually made, not the unavailable proposed PSF rewrite. |
| ZS-22, unequal bin axes | Already corrected in current upstream. Candidate corrects the fork's axes and WCS while preserving software-binning noise. |
| ZS-23, spectral extrapolation | Withdrawn by the report itself: extrapolation is part of the chosen synphot model. No automatic zero-fill was introduced. |

The extra PHOTLAM cube-unit claim was not investigated. Unrun witnesses and unavailable verification suites are not counted as passes.

The [resolved-input check](../validation/xshooter/data/representative_sampling.txt) uses the configured seeing PSF at 0.5701 um (FWHM 0.633165 arcsec), a 0.70-arcsec slit, and the report's independent Moffat quadrature. In-slit flux is 0.919472 of the analytic integral. This combines aperture rasterization, point placement and PSF sampling; it does not isolate an 8% PSF-normalization error. For Gaussian lines at 5600 Angstrom, seven sub-bin phases give prepared/exact integrated flux 0.999916–1.000086 at intrinsic resolving power 20,000, versus 0.002829–3.308268 at 200,000. Broad kilonova features do not inherit that worst-case line error automatically, but narrow sky and telluric structure still deserve convergence tests.

The candidate's six-channel line-variance tests and six-channel native-mapping photon tests pass with controlled, physically specified cubes. Its six dark-current noise tests also pass. Those isolated successes do not override the 9.5% full-scene grid dependence. The aperture/PSF pipeline still changes the illumination delivered to the mapper. **The candidates are not ready to become the science baseline.** Resolving this requires a bounded common rendering correction and independent checks, not a throughput renormalization.

## Regression results

The environment was the configured `zssim` Python 3.14.6 environment. Clean upstream was pinned to [ScopeSim `2d24516855e35bdffa5acd0485dbacd41855c727`](https://github.com/AstarVienna/ScopeSim/commit/2d24516855e35bdffa5acd0485dbacd41855c727) (0.12.1.dev6) and [IRDB `14d3078b7b3133cac3265b704304b932167edbef`](https://github.com/AstarVienna/irdb/commit/14d3078b7b3133cac3265b704304b932167edbef). Upstream was built into an isolated target so its version metadata was not confused with the installed fork. No user environment packages were replaced.

| ScopeSim offline suite (`-m 'not webtest'`) | Passed | Failed | Setup errors | Skipped | Expected failures |
|---|---:|---:|---:|---:|---:|
| Original fork | 1032 | 26 | 20 | 13 | 16 |
| Feature branch | 1053 | 26 | 20 | 13 | 16 |
| Candidate branch | 1058 | 26 | 20 | 13 | 16 |
| Clean current upstream | 1028 | 0 | 0 | 17 | 16 |
| Limited upstream port | 1035 | 0 | 0 | 17 | 16 |

There are 46 deselected web tests in the fork and 42 upstream. The exact 46 inherited failing/error case identifiers match across all three fork runs; [the list](../validation/xshooter/data/inherited_test_failures.txt) is retained. Many share an unresolved `!TEL.temperature` configuration failure; others involve projection, effect ordering, trace-coordinate inspection and an echelle clipping assertion. These are not newly introduced failures or a passing fork suite. The feature tests add 21 passes. IRDB X-shooter has 26 passing tests; candidate X-shooter plus ZShooter physical tests have 44 passes. The zs-scopesim source tests have seven passes and the same two pre-existing lamp-normalization and zero-redshift-ordering failures.

The [limited port](../validation/xshooter/upstream-port.patch) contains detector-arm gain/mode/ID/header handling, explicit cached-sky units, and WCS/local-aperture coordinates including METIS slicing. It does **not** port the complete X-shooter fork additions, the analytical ZShooter echelle implementation, or the report's proposed PSF renderer. Passing upstream tests therefore demonstrates compatibility of that subset only.

The real IRDB instrument suite covered MICADO compilation and spectroscopy tests, METIS, MOSAIC, and HAWKI: clean upstream had 461 passes, four skips, and three expected failures. The port initially had four MOSAIC failures caused by concurrent SkyCalc cache writes; all four passed when rerun serially. All 461 cases consequently passed, but the four MICADO spectroscopy skips remain skips. [Raw suite counts](../validation/xshooter/data/test_summary.csv) retain the initial failures and rerun separately.

The [bounded observation notebook](../validation/xshooter/Upstream_Observations.ipynb) supplements those tests with actual empty-sky and point-source observations through ten configurations. It keeps optical effects enabled, uses narrow wavelength intervals to bound memory, and requires finite arrays and positive source counts. MICADO spectroscopy uses the native full focal plane because the chosen orders miss the default central detector window. These are image-plane electron rates before stochastic detector effects, not complete detector/readout or full-band notebook tests.

Imaging (MICADO SCAO/MCAO, METIS LM/N, HAWKI) and MOSAIC's tested MOS-LR-R mode are unchanged pixel for pixel. Spectroscopic coordinate corrections change profiles: MICADO SPEC total source rate changes by -0.0021%, METIS LSS-L by +0.021%, LSS-N by -0.902%, and LMS by -1.002%. The largest centroid change is about 0.90 image-plane pixel; LMS's projected RMS width changes by about 1.50 pixels. [Full flux, position, width and pixel-difference measurements](../validation/xshooter/data/upstream_observations.csv) are saved. These differences need an independent optical expectation before adopting the coordinate port; preserving total counts is insufficient. The original upstream documentation notebooks were not all executed unchanged.

## Independent efficiency check

[ESO describes its efficiency QC](https://www.eso.org/observing/dfo/quality/XSHOOTER/qc/standard_qc1.html) as extinction-corrected telescope+instrument+detector efficiency measured from standards. The [public QC table](https://archive.eso.org/bin/qc1_cgi?action=qc1_browse_table&table=xshooter_fluxeff) was queried for 2026-08-01 through 2026-10-16 and for the 2023-12-31 civil night containing the January 1 calibration exposure. The recent query returned 81 rows, 27 per arm, ending with civil night 2026-09-21. It provides no October coverage in the returned sample.

The reference rows identify the same January 2024 A-nod raw files used in the calibration, so they are an epoch anchor, not an independent validation set. The recent observations are independent exposures. All use a five-arcsec slit and all have `photometric=N`; this does not prove they were all cloudy, but prevents treating the sample as a selected photometric absolute standard. The model was not refitted.

| Arm | QC interval, nm | 2024 efficiency | 2026 median | Median / 2024 |
|---|---|---:|---:|---:|
| UVB | 450–470 | 0.220276 | 0.216822 | 0.9843 |
| UVB | 510–530 | 0.196738 | 0.192457 | 0.9782 |
| VIS | 672–680 | 0.217513 | 0.211443 | 0.9721 |
| VIS | 745–756 | 0.224548 | 0.218945 | 0.9750 |
| VIS | 992–999 | 0.046265 | 0.045320 | 0.9796 |
| NIR | 1514–1548 | 0.208656 | 0.226123 | 1.0837 |
| NIR | 2214–2243 | 0.205578 | 0.227726 | 1.1077 |

The [2024 export](../validation/xshooter/data/qc_2024_reference.txt), [2026 export](../validation/xshooter/data/qc_2026.txt), and [percentile comparison](../validation/xshooter/data/qc_comparison.csv) support retaining the optical calibration for a first pass in these intervals, with uncertainty from scatter and observation conditions. They do not validate narrow-slit losses, backgrounds, telluric intervals, host-galaxy projection, or the common engine's absolute flux scale. NIR still uses the ETC approximation, so its 8–11% observed epoch increase is **not** a measured simulation residual. A future absolute QC-to-model comparison must also reconcile collecting-area conventions: the ESO pipeline efficiency calculation uses 51.2 m2, while the configured UT3 area is about 51.8 m2. The epoch ratios here use the same QC convention on both sides.

## Repeating the checks

Use the configured interpreter. The commands below show `conda run -n zssim`; the review used its resolved Python executable. Keep feature and candidate checkouts separate and keep calibration files fixed. In the respective ScopeSim checkout, run:

```bash
conda run -n zssim python -m pytest scopesim/tests -m 'not webtest'
```

In IRDB, with `PYTHONPATH` selecting the matching ScopeSim checkout:

```bash
conda run -n zssim python -m pytest XSHOOTER/tests
# Candidate checkout only:
conda run -n zssim python -m pytest XSHOOTER/tests ZShooter_v2/tests/test_fidelity_candidates.py
```

For upstream, create checkouts named `upstream-scopesim`, `upstream-port`, and `upstream-irdb` under `/tmp/xshooter-review`, at the pinned revisions above. Apply `validation/xshooter/upstream-port.patch` to `upstream-port`. Build the upstream package into a separate target without changing the active environment:

```bash
conda run -n zssim python -m pip install --no-deps --target /tmp/xshooter-review/upstream-runtime /tmp/xshooter-review/upstream-scopesim
```

Select each source checkout ahead of that target on `PYTHONPATH`. From the upstream IRDB checkout, run:

```bash
conda run -n zssim python -m pytest MICADO/test_micado/test_micado_compiles.py MICADO/test_micado/test_micado_spec.py METIS/tests MOSAIC/tests HAWKI/test_hawki/test_full_package_hawki.py
```

Run the baseline and port sequentially, or give them separate `SKYCALC_IPY_CACHE_DIR` values. In `validation/xshooter/Upstream_Observations.ipynb`, run first with `engine='upstream-scopesim'`, restart the kernel, then run with `engine='upstream-port'`. The final cell requires both sets of saved arrays. The notebook does not install software or launch subprocesses.

The report witnesses require the separately supplied `yashvi_demos` folder. Select `PYTHONPATH` and `IRDB`, then run `validation/xshooter/run_report_witness.py` with one witness path. `representative_sampling.py` takes the folder path and prints the ordinary-seeing and resolved-line comparisons. The report inputs are not runtime dependencies and are not redistributed here.

The saved QC exports can be recomputed without network access:

```bash
conda run -n zssim python validation/xshooter/compare_qc.py validation/xshooter/data/qc_2024_reference.txt validation/xshooter/data/qc_2026.txt validation/xshooter/data/qc_comparison.csv
```

To refresh the sample, request `https://archive.eso.org/qc1/qc1_cgi` with `action=qc1_browse_get`, `table=xshooter_fluxeff`, `from`/`to` dates, `output=ascii`, and `field_NAME=NAME` for the columns in the saved export. Preserve the raw response and its date range; a new calibration fit must use a different held-out validation set. A close X-shooter fit can support the comparison only to the extent that the tested physical and numerical paths are shared and independently checked.
