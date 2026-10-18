# ZShooter analytical geometry correction

**The analytical simulator now uses the archive-derived grating settings and detector rotations, and centers the complete FSR/slit envelope.** The detector-filling angle estimate is no longer used for ZShooter_v2. These changes live on `fix/zshooter-xdisp-geometry` in each repository and have **not** been merged into `xshooter`.

## Adopted settings and resulting footprints

Source: `20260610B_ZS_6CH_Complete_Spectrograph.zar`. The VIS echelle ruling is now 50.15 grooves/mm; NIR remains 24. CX rulings are 2000/1100/625/350/300/270 grooves/mm. The configured camera and effective dispersion focal lengths remain 225/270 mm, with 15 micrometer pixels and a 10 arcsecond slit.

| Channel | Orders | CX incidence/central angle (deg) | Detector-point rotation (deg) | Full FSR/slit envelope, X by Y (mm) | Mean order tilt (deg) |
|---|---:|---:|---:|---:|---:|
| B | 85-118 | 21.20 | -9.50 | 22.14 x 59.05 | -1.08 |
| G | 56-85 | 16.75 | -7.00 | 25.29 x 59.03 | -0.46 |
| R | 37-56 | 14.10 | -6.00 | 32.95 x 50.16 | -0.46 |
| YJ | 57-80 | 10.90 | -5.00 | 21.15 x 31.97 | -0.67 |
| H | 42-51 | 13.70 | -6.10 | 28.89 x 23.44 | -0.64 |
| K | 31-38 | 16.80 | -6.50 | 39.50 x 30.15 | +0.20 |

Within the planar unfolding, CX incidence is $\alpha=-s\theta_{85}$, with propagation sign $s$ negative for B/G/YJ/H and positive for R/K. Detector point rotation is $-\theta_{166}$ because [coordinate breaks rotate the axes](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v26102/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Coordinate_Break.html). The [extracted prescription](../validation/zshooter_xdisp/data/zemax_channels.csv) retains the original signed tilts and camera space. No rotation search is applied.

![Previous and adopted analytical footprints](../validation/zshooter_xdisp/data/detector_footprints.png)

Upper row: preceding feature-branch prescription/placement, including the corrected scalar angle equation. Lower row: adopted settings and centering. Colored curves are complete FSR centerlines; gray curves are slit ends, all from the simulator's actual mapping. Mean tilt weights orders equally. CSVs give [per-order values](../validation/zshooter_xdisp/data/orders.csv) and [boundary margins](../validation/zshooter_xdisp/data/footprints.csv); negative margins mean analytical overhang.

Solid boundaries: configured 30.720 x 61.440 mm VIS and 30.720 mm square NIR detectors. Dashed boundaries: the archive's 36.864 mm square NIR aperture, centered solely to compare sizes. This corresponds to 18 micrometer pixels at 2048 pixels; the simulation remains at 15 micrometers.

**Remaining discrepancy:** R is 32.95 mm wide against 30.72 mm; K is 39.50 mm wide, exceeding even the archive aperture. YJ is 31.97 mm high, slightly beyond the configured detector. B/G/H fit. R/K order limits agree with the archive; their excess analytical widths remain a separate investigation. Camera off-axis projection, distortion and non-coplanar coupling remain outside this correction.

## What changed in the calculation

ScopeSim centers all configured FSRs and slit endpoints, sampled once at 2049 wavelengths per order, then projects the translated detector rectangle back onto the internal dispersion axis to locate intersections. Translation preserves dispersion and slit extent. Simulation traces can extend beyond an FSR where admitted by the detector; footprint assessment uses full FSRs. No new public API is required.

The existing efficiency effect reads the same prescription. Its VPH approximation uses $\lambda_{\mathrm{peak}}=2d\sin\alpha$ for an explicit angle, shifting the efficiency by -4.1 to +5.6 percentage points across the six nominal bandpasses. The VIS ruling change also shifts order/blaze centers by -0.299%; fixed-wavelength comparisons therefore sample different positions on the blaze. [Efficiency samples](../validation/zshooter_xdisp/data/efficiency.csv) give both prescriptions at the adopted FSR endpoints and centers, separating VPH from combined grating efficiency. No efficiency renormalization was added.

## Validation and runtime

**49 targeted tests pass**, covering angle signs, detector rotation, detector-size independence, centering and translated clipping. The six-channel assessment verifies archive agreement, matching trace/efficiency optics, all 126 orders and convergence against 8193 samples per order: centering changes by less than 0.00001 pixel against a 0.01-pixel tolerance. Detector padding/height changes preserve explicit optics and VPH efficiency. The figure was visually inspected.

| Measurement | Previous | Adopted |
|---|---:|---:|
| Trace construction, median of three | 0.90 s | 0.95 s |
| Full-band `observe()` | 47.96 s | 48.17 s |
| Initial train setup, including ESO request | 77.61 s | 8.05 s |

The AB=20 flat-continuum observation uses all six full-size image planes and the 0.7 arcsecond slit. All images are finite and nonempty; detector readout is not timed. Setup includes external SkyCalc latency; `observe()` differs by 0.4%. Raw timings: [previous](../validation/zshooter_xdisp/data/timings_previous.csv), [adopted](../validation/zshooter_xdisp/data/timings_adopted.csv). The separate assessment/plot takes about five seconds.

Separate existing issue encountered during setup: the feature-branch base's brightness-only configuration repeatedly resolves `Time.now()`, causing its PSF cache to compare incompatible AltAz frames. Both timing runs use `!OBS.mjdobs = '2025-01-01T10:00:00'`. That PSF behavior is outside this geometry patch.

## Review and reproduce

Commits: ScopeSim `87d6f581` (placement/tests), IRDB `9fa8317` (prescription), with this report/checks committed separately in zs-scopesim. The independent branch bases remain ScopeSim `main` and IRDB `dev_master`.

From the zs-scopesim feature worktree, with the sibling feature checkouts available:

```sh
git -C ../scopesim show aefbe60d:scopesim/effects/spectral_trace_list.py > /tmp/zshooter-before-traces.py
git -C ../irdb show cf9110a:ZShooter_v2/traces/echelle_trace_parameters.dat > /tmp/zshooter-before-parameters.dat
PYTHONPATH=../scopesim conda run -n zssim python validation/zshooter_xdisp/geometry.py \
  --parameters ../irdb/ZShooter_v2/traces/echelle_trace_parameters.dat \
  --baseline-parameters /tmp/zshooter-before-parameters.dat --baseline-traces /tmp/zshooter-before-traces.py \
  --zemax-parameters validation/zshooter_xdisp/data/zemax_channels.csv --output /tmp/zshooter-geometry-review
PYTHONPATH=../scopesim:. conda run -n zssim python -m pytest tests/test_xdisp_geometry.py tests/test_zemax_prescription.py \
  ../scopesim/scopesim/tests/tests_effects/test_SpectralTraceList.py ../scopesim/scopesim/tests/tests_optics/test_echelle.py -q
PYTHONPATH=../scopesim conda run -n zssim python validation/zshooter_xdisp/benchmark.py \
  --irdb ../irdb --atmosphere-data /Users/jibailey/src/irdb/MaunaKea --output /tmp/zshooter-timings.csv
```

The worktrees are under `/private/tmp/zshooter-xdisp/`; the installed environment and active notebooks are unchanged. The benchmark needs the existing TAPAS data and ESO access. Archive re-extraction uses [zemax_prescription.py](../validation/zshooter_xdisp/zemax_prescription.py) with `zmxtools`, the archive path and `--output` directory. `zmxtools` is not a simulator dependency.
