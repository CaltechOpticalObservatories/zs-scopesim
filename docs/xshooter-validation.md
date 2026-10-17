# X-shooter and shared simulator validation

Assessment updated 17 October 2026. The shared numerical changes are now adopted on the **`xshooter` branches in all three repositories**. Upstream ScopeSim and IRDB have been merged, with the instrument additions retained. Nothing has been pushed.

Start with [the flat-AB comparison notebook](../notebooks/02_Flat_AB_Comparison.ipynb) and [its executed results](xshooter-comparison-results.md). Its first source is a centered AB 20 flat continuum, compatible with the technical normalization in `zshooter_performance.ipynb`. [Instrument inputs and literature](xshooter.md) describe the X-shooter prescription. [The earlier assessment](xshooter-validation-2026-10-16.md) is a historical record of the pre-adoption findings; its candidate status and test counts are superseded here.

## What the comparison establishes

The same engine now integrates the effective PSF over physical slit cells, admits light from outside the slit, preserves native spectral knots until flux integration, and maps through physical trace coordinates for both instruments. Those improvements strengthen a shared numerical comparison. They do not make an X-shooter calibration fit an independent validation of ZShooter's design assumptions.

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Analytic slit integrals, spectral photon integrals, detector noise and WCS tests | Specified numerical relationships on shared code paths | A complete optical or detector model |
| X-shooter trace, response, LSF and detector inputs | A literature/calibration-based instrument prescription | Independent accuracy when tested against the same fitted inputs |
| Held-out QC efficiency samples | Broad consistency of the optical response and observed epoch variation | Narrow-slit sensitivity, sky subtraction, or NIR ETC-model accuracy |
| Flat-AB notebook, native line probes, common-resolution extraction | A reproducible forecast under explicit common conditions | On-sky limiting magnitudes including reduction systematics |
| ZShooter detector and optical inputs | The configured engineering design point | Measured properties of hardware that does not yet exist |

The PSF is an **effective** PSF. Bounded wavelength averaging, finite support and detector quadrature remain deliberate approximations. We have not implemented a full sky+telescope+instrument wavelength-dependent convolution, nor Pyxel detector statistics. These are modelling choices whose adequacy depends on the science case, not blanket software defects.

## Adopted findings and remaining approximations

| Finding | Current treatment |
|---|---|
| ZS-01/06: repeated transmission and airmass | Previously corrected; retained through the upstream merge. |
| ZS-02: rasterized slit width | Physical aperture edges are tiled exactly. The 0.30, 0.33 and 0.70 arcsec regression cases no longer substitute an integer-pixel slit width. Detector pitch is independent of this integration grid. |
| ZS-03/04: point placement and sampled Moffat | A continuous point position is integrated directly over each slit cell; no extra bilinear deposition broadening is introduced before the effective Moffat PSF. Quadrature tests include 0.10, 0.14 and 0.70 arcsec FWHM and off-center positions. |
| ZS-09/10: resolved inputs and off-slit light | PSF support is extracted before aperture clipping. Image surface brightness uses pixel overlaps and signed WCS scales; image/cube/point normalization is tested. Large cubes are processed in wavelength chunks. Rotated-image reprojection retains an affine interpolation approximation. |
| ZS-11/12: slit coordinates and ends | Actual WCS coordinates and physical slit endpoints are used. IFU slices subtract their own physical center. Spectral padding includes the slit shift and configured optical LSF before clipping back to the requested band. |
| ZS-13/14: deposition and edges | Integer rounding and boundary masking are corrected for the remaining deposition path; edge points do not wrap to the opposite edge or index outside the array. |
| ZS-07: narrow spectral structure | Native piecewise-linear spectra and cubes are integrated over FOV wavelength-bin edges. Early point resampling is removed. This preserves unresolved line photons represented in the input; it cannot recover a line absent from the input grid. |
| ZS-16/19: grid scale, Jacobian and dispersion | Detector area comes from the physical detector WCS determinant. Cross-slit dispersion comes from the configured trace. ZShooter uses its analytical optical prescription with an explicit equal along/across-slit magnification assumption. All six channel tests exercise three sky grids and native detector mapping. |
| ZS-22: unequal and charge binning | Upstream axis correction is retained; common header/binning geometry is used in readout and coordinate inspection. CCD charge binning precedes read noise; software binning remains after it. Both instruments support independent visible-arm charge bins. |
| Exposure settings and read noise | Shared selectors retain independent per-arm DIT/NDIT, gain, noise and metadata. ZShooter now uses the same charge-binning mechanism. The NIR RON table is a generic reusable effect; another design can supply a specification, estimate, design point or measured table. |
| Dark current and repeated readouts | ZShooter dark current participates in shot noise. A new readout advances the random realization; an explicit readout ID reproduces it. |
| ZS-05/17/18/20/21 | Finite sampling, PSF support and discrete external PSF models remain approximations. The adopted spatial changes are targeted at the effective Moffat spectroscopy path; we have not rewritten every imaging PSF. |
| ZS-15: ADC | Fast integer shifts retain their quantization. A negative-shift boundary error is fixed. The existing subpixel interpolation path caches separable spatial operators and does not mix wavelength planes. |
| ZS-23: extrapolation | Input spectral-model extrapolation remains its declared convention. Trace interpolation uses zero outside its prepared support, with physical spectral padding to avoid artificial band-edge losses. These are distinct operations. |

Pre-broadening an input to instrument resolution can reduce sampling errors for a smooth continuum. It does not correct slit geometry, off-slit illumination, or resolved-source normalization, and it can double-count a simulated LSF. Native knots plus bin integrals address narrow lamps and airglow without requiring such pre-smoothing. The configured spectral grid still needs to resolve the *instrument-broadened* output profile.

The candidate physical-coordinate and Jacobian work has therefore been adopted together with the spatial and spectral fixes, rather than treating the earlier full-scene 9.5% grid difference as a successful convergence result. The old 45%/9.5% figures belong to the historical branches and remain in the archived assessment. Controlled six-channel normalization checks agree within their 1% numerical test tolerance. A separate six-channel **observed point-source** test changes the actual spatial grid by factors of two and four while keeping detector pitch fixed: total source rates change by less than 0.1% relative to the finest grid. [Measurements](../validation/xshooter/data/observed_grid_convergence.csv) use 0.70 arcsec seeing/slit, an offset source, prescribed optics and narrow wavelength intervals; sky and ADC are excluded to isolate spatial integration. This is not a universal 0.1% instrument-accuracy claim.

## Extraction and comparison conventions

Both trains use the same evaluated TAPAS transmission, sky continuum and PALACE airglow curves, with airmass 1.3, seeing 0.7 arcsec, a fixed epoch and one hour on source (300 s x 12). Telescope area, optics, slit losses, ADC and detector specifications remain instrument-specific. This is an instrumental comparison under a common sky, not a comparison of observing sites.

The default X-shooter setting is 1.0/0.9/0.9 arcsec, slow UVB/VIS CCD readout, spectral 1x2 charge binning, unbinned NIR. The optional higher-resolution setting is 0.8/0.9/0.6 arcsec with unbinned CCDs, nominal manual R=6700/8900/8100. These are explicit manual-based choices; neither is claimed to reproduce the exact hardware setup in Pian et al. ZShooter uses 0.70 arcsec slits and its current qCCD/APD specifications.

The notebook extracts integer spatial boxcars on native pixels. Nine samples per order select the smallest width within 0.01 mag of the best expected 5-sigma limit. There is no fitted spatial profile, Horne extraction or ESO reduction-pipeline dependency. Source, background, dark and read variances are counted in accumulated electrons. Equal-time nod subtraction doubles background variance, not source shot noise. CCD sum and NIR average output conventions are preserved in the actual FITS frames.

Line probes measure native FWHM and second moment. Gaussian degradation and bin integration carry `W diag(V) W.T`; each nominal resolution element sums four output samples including their covariance. Neighboring resolution elements remain correlated. Overlapping orders are handled conservatively by selecting the single best order, without assuming a coaddition gain. OH wavelength selection is a separate diagnostic with its retained-flux fraction and irregular spectral response stated explicitly. The technical curves always start with flat AB; G5V and kilonova source recipes provide optional science context. Independent noisy source/sky frames use distinct readout IDs; the notebook also checks their extracted residuals against the forecast variance.

ZShooter's diffuse-emissivity option remains disabled as in its current performance notebook, while X-shooter's empirical NIR background remains enabled. Detector systematics and imperfect sky subtraction are omitted. These are visible model-input differences to revisit in engineering comparisons, not compensating corrections hidden in the shared engine.

## Runtime and reproducibility

[Performance.ipynb](../validation/xshooter/Performance.ipynb) and [saved timings](../validation/xshooter/data/full_frame_timings.csv) record full-frame observations and separate setup/readout costs. Final runs used fresh kernels and wall-clock profiling on this workstation:

| Instrument | Before shared work: observe | Adopted: observe | Adopted: readout | Adopted process peak |
|---|---:|---:|---:|---:|
| ZShooter, six detectors | 69-70 s | 60.9 s | 1.7 s | 5.1 GiB |
| X-shooter, three detectors | 590.3 s | 74.4 s | 0.9 s | 9.3 GiB |

Profiler overhead is included. The historical X-shooter 35.7 GiB peak is cumulative after earlier runs, so it is not a clean isolated-memory comparison. Network time is separate: the first historical ZShooter setup included a SkyCalc request. X-shooter benchmarks use the same cached ESO query (airmass 1.3, PWV 2.5 mm, R=200000); the cache is optional and untracked. The comparison notebook times its separate common-sky observations too: changing the complexity of the sky model can change runtime.

The main savings are integrating point PSFs without giant padded cubes, reusing native spectrum knots and rectified wavelength interpolation, caching atmospheric spatial-shift operators, and releasing input cubes after mapping. No five-minute per-observation computation is required by the default comparison. The notebook performs multiple observations and line probes; its total runtime is not the runtime of one simulation.

## Tests, skips and upstream integration

The adopted ScopeSim optics/effects/detector/source/utility suites plus X-shooter and six-channel ZShooter physical tests report **1,140 passed, 5 skipped, 27 web tests deselected and 16 expected failures**. Slow-marked tests were included. This count is for the named suites, not a claim that every test or documentation notebook in every repository passed.

Two FOV cross-representation tests are skipped because they call a single pre-refactor FOV object as multiple output types; they require distinct FOV subclasses after the upstream split. The old cube/image flux expected failures also use obsolete `fov.spectra`, inconsistent unweighted sums or historical edge corrections. The new point/image/cube integration, signed-grid, off-slit and narrow-line tests exercise the relevant spectroscopy relationships. They do not clear the unresolved issues in every other FOV output representation. These tests are not excluded for being expensive. The [test audit](../validation/xshooter/data/shared_test_audit.txt) lists inherited skips and expected failures, including the discrete-PSF center case.

The merged engine passes **461 MICADO/METIS/MOSAIC/HAWKI package tests**, with four skips and three expected failures. This run exposed and resolved single-plane fibre-layout routing, METIS scalar-transform compatibility, and applying slit padding to a fibre-bundle FOV. A subsequent real LMS observation exposed missing inherited trace parameters; that correction passes 39 targeted METIS/trace tests. These 39 overlap the broader suites and are not an additional independent test count.

[Upstream_Adoption.ipynb](../validation/xshooter/Upstream_Adoption.ipynb) executes ten bounded observations on the adopted stack: MICADO SCAO/MCAO imaging and spectroscopy; METIS LM/N imaging, LM/N long-slit spectroscopy and LMS; MOSAIC MOS; and HAWKI. Every case produces positive source flux, finite image planes and finite checked readouts. The [saved results](../validation/xshooter/data/adopted_upstream_observations.csv) include the usually skipped MICADO spectroscopy path with a narrow wavelength interval and its real full detector layout. These are execution/normalization checks, not external sensitivity validations. The earlier raw observation results and patch remain historical evidence for the initial limited port.

The workspace test run reports **103 passes and 16 failures**. The six new extraction/covariance tests pass. Two existing source tests disagree with lamp-amplitude handling and zero-redshift validation order; 14 existing validation-helper tests use stale interfaces, missing `AOEnhanceablePSF` fixtures or obsolete mocked PSF signatures. The source/helper implementations and those tests are unchanged from the workspace's original HEAD. They are listed in the audit, not hidden or counted as passing. Updating that separate test/implementation contract is outside this patch.

From the ScopeSim checkout, with IRDB at its sibling path:

```bash
conda run -n zssim python -m pytest scopesim/tests/tests_optics scopesim/tests/tests_effects scopesim/tests/tests_detector scopesim/tests/tests_source scopesim/tests/test_utils_functions.py ../irdb/XSHOOTER/tests ../irdb/ZShooter_v2/tests/test_fidelity_candidates.py -m 'not webtest'
```

Use the environment manifest to reproduce the adopted stack. ScopeSim now follows upstream's Python >=3.12,<3.15 requirement; the configured environment is Python 3.14.6. The editable install points at the actual ScopeSim checkout, not a temporary review tree.

| Repository | Adopted tip | Commit organization |
|---|---|---|
| ScopeSim | `06f64b092eab01ebe4da789149d1a3be514c17b4` | Upstream merge; physical coordinates; caches; spectral integrals; exposure metadata; integrated PSF; cached shifts; spectral edge padding; binning/readout; coordinate inspection; upstream fibre/METIS compatibility |
| IRDB | `fc3e2b9a2824bf57b6e1aa6dd33ffdc26229be00` | Upstream merge; physical ZShooter dispersion and dark noise; independent CCD charge bins; six-channel observed grid convergence |
| zs-scopesim | `xshooter` | Stack adoption and centered-source extraction; comparison/forecast code; executed assessment and performance artifacts |

## Independent checks and next uses

The independent efficiency comparison is sufficiently close for this first-pass objective; no response refit was made. The roughly 1.2% ESO collecting-area convention difference is a release note for an ESO-facing package, not a blocker. The saved 2024/2026 QC comparison shows optical epoch changes of about 1.6-2.8% and NIR changes of 8-11%; the latter are observed epoch differences, **not** measured NIR simulation residuals. The NIR prescription still uses an ETC-based response approximation.

For a claim about absolute on-sky sensitivity, the next evidence should be an independent observation with recorded seeing, slit, binning, DIT/NDIT, airmass, flux calibration and extraction convention, reduced with a comparable boxcar. Hold that observation out from calibration. Compare counts, background, line width and uncertainty separately before comparing limiting magnitude. An ESO/Reflex or PyReduce extraction is useful as a later cross-check, but is not required to make this transparent boxcar forecast useful.

The [ESO manual](https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_P114v1.pdf) supplies instrument settings. [Pian et al. (2017)](https://arxiv.org/abs/1710.05858) supplies the motivating kilonova context and a reduction comparison target. A future kilonova run should use its documented observing setup or state a deliberately chosen alternative; the flat continuum remains the technical baseline.
