# ZShooter detector operating points

`SPEC` and `SPEC_ALTDETECT` now select complete detector packages in IRDB. The alternative uses untapered CCDs in B/G/R, the same LmAPD in YJ, and 18 um H2RGs in H/K. QE, physical pitch, dark current, effective read noise and usable well come from that configuration. The comparison simulates and extracts each package instead of rescaling the original detector samples by a QE ratio.

```python
commands = scopesim.UserCommands(use_instrument="ZShooter_v2", set_modes=["SPEC_ALTDETECT"])
train = scopesim.OpticalTrain(commands)
```

Construct a new train when changing detector packages: pitch and QE enter effect initialization. Slit/exposure changes within a prepared package retain the existing ScopeSim update/readout interface.

## Configured engineering points

| Package and channels | Detector | Pitch [um] | Effective RN [input e RMS/integration] | Dark [input e/hour/pixel] | Photon variance factor F | Avalanche gain M | Usable input well [e/integration] |
|---|---|---:|---:|---:|---:|---:|---:|
| SPEC B/G/R | tapered qCCD | 15 | .1 | .5976 | 1 | 1 | 60,000 |
| SPEC YJ/H/K | LmAPD | 15 | .5 | .36 | 1.2 | 10 | 15,000 |
| SPEC_ALTDETECT B/G/R | untapered CCD | 15 | 2.5 | 2 | 1 | 1 | 60,000 |
| SPEC_ALTDETECT YJ | LmAPD | 15 | .5 | .36 | 1.2 | 10 | 15,000 |
| SPEC_ALTDETECT H/K | H2RG | 18 | 6 | 23 | 1 | 1 | 60,000 |

These are design inputs, not measured vendor specifications. Per-arm properties live in `ZShooter_DET.yaml` and `ZShooter_ALTDETECT.yaml`, including `!DET.usable_well_h`, `!DET.ron_h`, `!DET.dark_h`, `!DET.excess_noise_factor_h` and `!DET.pixel_size_h`. The qCCD low RN remains a specified effective operating point; its sampling time has not been specified.

Pixel counts remain 2048 x 4096 in B/G/R and 2048 x 2048 in YJ/H/K. An H2RG therefore covers 36.864 mm rather than 30.72 mm in each direction. Physical optics, order numbers and offsets stay fixed. H/K gain focal-plane coverage and use fewer detector pixels per physical resolution element. YJ retains its omitted red order. Degree-six trace fits with seven slit samples keep the mapping consistent over the larger coverage. Tests compare all 125 orders at common physical coordinates between pitches, to 0.01 of a 15 um pixel.

## Explicit exposure settings

Each notebook defines exposure settings near the beginning. NDIT counts independent integrations or ramps, not samples within a ramp. There is **no automatic increase of NDIT**, adjustment of DIT, or saturation clipping. Readout checks expected source + optical background + dark charge per integration, after CCD charge binning, and warns if it exceeds the configured usable limit. `PEAKDIT` and `WELLDEP` record the check in the image header. The accumulated image may exceed one well when each contributing integration is below its limit.

| Package/channel | Five minutes | One hour |
|---|---|---|
| Either package, B/G/R | 300 s x 1 | 1200 s x 3 |
| Either package, YJ LmAPD | 100 s x 3 | 100 s x 36 |
| SPEC H LmAPD | 30 s x 10 | 30 s x 120 |
| SPEC K LmAPD | 60 s x 5 | 60 s x 60 |
| SPEC_ALTDETECT H H2RG | 75 s x 4 | 90 s x 40 |
| SPEC_ALTDETECT K H2RG | 150 s x 2 | 150 s x 24 |
| X-shooter UVB/VIS | 300 s x 1 | 1200 s x 3 |
| X-shooter faint NIR | 300 s x 1 | 300 s x 12 |
| X-shooter high NIR | 300 s x 1 | 450 s x 8 |

Kilonova science, standard and calibration DITs are likewise explicit. Their sky/scene differs from flat AB, so their own peak tables are the relevant checks. An extraction mask cannot make the incoming OH peaks safe for the detector.

The flat-AB wide-slit SPEC H/K peaks are 11,442/12,341 e per integration (76.3%/82.3% of usable well). Alternative H2RG H/K peaks are 49,412/43,725 e (82.4%/72.9%). YJ peaks at 6,962 e in both packages. [The peak table](../validation/xshooter/data/comparison/saturation.csv) records both durations and all slit/seeing cases. These replace the earlier operationally infeasible 1800-second NIR forecasts.

X-shooter CCD checks convert the manual's 65,000 ADU ceiling using the configured gain; NIR retains the adopted conservative 89,000-electron limit. Native NIR outputs average NDIT, while CCD outputs sum them; prepared reduction inputs convert the actual outputs to accumulated electrons. Sources: [ESO manual, Table 9](https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_P114v1.pdf) and [NIR readout note](https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/reportNDreadoutpublic.pdf).

## Physical QE and avalanche variance

The historical `APD_TER.dat` equals `H2RG_TER.dat / 1.2`. That was a photon-noise sensitivity approximation, not an independently specified APD QE. Both packages now use `H2RG_TER.dat` as the common **assumed physical NIR QE**, with F=1.2 applied to LmAPD photon variance. This inherited estimated curve needs a vendor-specific replacement when available. YJ's detector and QE are identical between packages.

Relative to the old proxy, LmAPD mean photon counts increase by 20%. Photon-dominated sensitivity is unchanged: physical signal and photon variance increase in the required ratio. Read noise is no longer indirectly penalized by reducing signal, and physical charge is available for the well check.

For accumulated source electrons S, optical background B, dark D, integration count N and effective input RN r, independent equal-time blank subtraction gives

\[
V = F S + 2\left(F B + D + N r^2\right).
\]

Optical thermal photons belong in B. Dark uses the specified independent input-referred mean/Poisson variance. Effective RN is already input referred and is not divided by M again. This preserves the operating distinction you described: downstream read noise is independent of avalanche multiplication, and graded detectors can leave dark signal largely unamplified. [Huber et al. (2026), Sections 2.2-2.3](https://arxiv.org/html/2610.00799v1#S2.SS2) describe those mechanisms. Our dark term remains a simple engineering noise prescription, without a separate layer-by-layer charge-generation model.

Science frames draw photon, dark and read fluctuations. The LmAPD photon distribution is compound Poisson/gamma, with input-referred mean mu and variance F mu; F=1 gives ordinary Poisson. This supplies the desired first two moments without modelling ramps. Performance expected-rate frames omit random draws; their extraction variance includes the same photon, dark and N-read terms.

ZShooter outputs stay in input-referred electrons. Header avalanche gain describes the operating point and is not a second signal multiplication. ADC conversion/bias have not been assigned vendor values. For output conversion g in output electrons/ADU, photon mean and variance would be M(S+B)/g and M^2 F(S+B)/g^2. Gain-one electron storage is not a measured electronics gain.

## Reduction and verification

The DRP receives an expected pixel-variance image and uses configured trace geometry and detector WCS. Kilonova calibration directories include the detector mode. Reuse also checks saved mode, detector dimensions and physical WCS; a mismatch requires rebuilding calibrations.

The alternative package uses the separately simulated 2 arcsec illuminated-slit trace frames in every channel. The untapered CCD response made the old full-slit-flat settings merge BLUE orders and split GREEN orders. Mode-specific trace settings recover the configured 34 BLUE, 30 GREEN, 20 RED, 10 H and 8 K orders without duplicates; YJ keeps its existing short-slit settings. BLUE retains all 34 orders when its trace threshold is varied from 30 to 300 electrons. These settings belong to the DRP's `settings_SPEC_ALTDETECT.yaml`, selected from the calibration frame's `DETMODE` header.

The current SPEC kilonova run passes its six flat-AB flux controls: recovered/input ratios are .9989, .9995, .9976, .9697, 1.0297 and 1.0151 in B/G/R/YJ/H/K. Independent blank differences have standardized standard deviations .9996–1.0004; illuminated source-minus-blank residuals give .9988–1.0012. The [saved control measurements](../validation/xshooter/data/detector_packages/spec_flat_ab_bands.ecsv) include propagated uncertainties.

The [executed alternative-package reduction notebook](../validation/xshooter/Detector_Package_Reduction.ipynb) simulates and reduces separate calibrations, a standard and a flat-AB source in all six channels. Its recovered/input ratios are 1.0043, .9984, .9951, 1.0021, .9890 and 1.0276; each is within 1.5 propagated standard deviations of unity. [Flux measurements](../validation/xshooter/data/detector_packages/alternate_flat_ab_bands.ecsv), [recovery plot](../validation/xshooter/data/detector_packages/alternate_flat_ab_recovery.png) and [per-integration charge checks](../validation/xshooter/data/detector_packages/alternate_exposures.csv) are saved separately.

Both packages recover 22 YJ orders: the short, partially covered blue order 80 remains outside the measured trace set. The other five channels recover all configured orders, giving 124 measured orders in each package. This is distinct from the 125-order optical-geometry test. The alternative's median absolute wavelength-fit residuals are .018–.032 native pixels across the channels; [calibration measurements](../validation/xshooter/data/detector_packages/alternate_calibration.csv) and [measured traces](../validation/xshooter/data/detector_packages/alternate_measured_traces.png) show the actual coverage.

One separate reduction failure remains explicit: the 20-day SPEC BLUE scene does not converge in the fitted-profile uncertainty calculation. The 1.5- and 7.5-day scenes extracted in all six channels. In the failed swath the fitted spatial profile has large cancelling positive and negative terms; increasing the uncertainty solver from 40 to 400 iterations did not resolve it. The detector patch does not replace that estimator with a fixed profile, change the random realization to avoid it, or assign a substitute uncertainty. This is a faint-source extraction question, not a failed detector well or photon-variance check.

The X-shooter rerun likewise completes all simulated frames and both modes' 1.5- and 7.5-day extractions, then stops at the 20-day faint-mode UVB uncertainty solve. Its independent-blank standardized deviations are .99936–1.00046; source-minus-blank deviations are .99769–1.00263. Maximum NIR charge is 72,769 e/integration in faint mode and 78,852 e in high mode, below 89,000 e. [Peak charges](../validation/xshooter/data/detector_packages/xshooter_peaks.csv), [exposure settings](../validation/xshooter/data/detector_packages/xshooter_exposures.csv) and [noise checks](../validation/xshooter/data/detector_packages/xshooter_flat_ab_pixel_noise.csv) record these detector checks.

Its **faint-mode NIR flux control fails**. The retained-window recovered/input ratio is 4692, dominated by localized spikes near 2.268 um; this is an invalid calibrated spectrum. The saved flat normalization has 212 samples with blaze exactly 1, including that region, and the standard's blaze-corrected counts jump there too. [Affected calibration samples](../validation/xshooter/data/detector_packages/xshooter_faint_unity_blaze_samples.csv) and the [unclipped recovery plot](../validation/xshooter/data/detector_packages/xshooter_faint_flat_ab_recovery.png) preserve the failure. The origin of those calibration values still needs repair; no samples were silently excluded to improve closure. The other five mode/arm controls recover 1.0069, 1.0036 (faint UVB/VIS), and 1.0002, .9939, .9851 (high UVB/VIS/NIR); [all values and uncertainties](../validation/xshooter/data/detector_packages/xshooter_flat_ab_closure.ecsv) are saved. The new three-epoch kilonova figures have not been regenerated. These two notebook runs are not counted as full scientific-validation passes.

Targeted tests cover avalanche mean/variance, independent arm RNG draws, unaffected dark variance, independent-read accumulation, electron/ADU and sum/average conversions, well warnings without parameter mutation, physical trace invariance and calibration reuse. Notebook reruns supply full-frame peak and extraction checks.

## Memory during validation

The 2026-10-20 concurrent-run snapshot found ten kernels on the project server (port 8888), with about 86 GiB of macOS-reported footprint on the 64 GiB workstation. SPIE used 29.1 GiB, the active performance rerun 23.0 GiB, X-shooter kilonova 11.3 GiB, alternate-detector reduction 6.9 GiB and flat AB 6.1 GiB. Five idle kernels retained another 9.2 GiB. The DRP server on port 8889 had no sessions. [All kernel measurements](../validation/xshooter/data/detector_packages/kernel_memory.csv) include resident size and peak footprint.

Resident RAM alone substantially understated these allocations because macOS had compressed or paged much of them out. The system reported warning-level pressure. Over 63 seconds, swap writes averaged 47.5 MiB/s and compression processed about 433 MiB/s; swap use reached 42.5 GiB. These system-wide rates demonstrate active pressure, rather than merely an old swap file. Running several full validations concurrently was a poor scheduling choice. Heavy notebooks should run serially, and completed validation kernels should be closed after saving outputs. The active user performance run is left intact. The full per-case analysis tables also merit a focused memory review before expanding the trade grid further.

After saving and closing the flat-AB, SPIE, alternative-reduction and X-shooter validation kernels, system pressure returned to normal. At 22:04 UTC, the [remaining six kernels](../validation/xshooter/data/detector_packages/kernel_memory_after.csv) comprised the active performance notebook (13.3 GiB footprint) and the five idle notebooks (about 9.3 GiB). Swap use had fallen to 17.3 GiB. Remaining swap allocation need not immediately disappear after pressure falls. None of the user's existing idle kernels or active performance kernel was stopped.

For vendor discussions, specify physical QE; bias/temperature and M; F explicitly as a variance multiplier; effective RN for the completed integration; dark mean/variance in stated input/output units; usable well at that gain and accepted nonlinearity; ADC conversion/range; and reset/read overheads. CCD charge-binning limits and qCCD sampling time also need specifications. Until overheads are specified, reported durations are on-source integration time. An independent equal-time blank adds the same sky integration time before overheads.
