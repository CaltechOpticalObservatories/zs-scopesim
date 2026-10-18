# ZShooter cross-disperser geometry assessment

The automatic rotation tuning has been withdrawn and the original detector rotations restored. Its objective rewarded in-band coverage while allowing tilted and missing orders. That was the wrong objective: the intended adjustment reduces average order tilt, with the complete FSRs and slit ends included in the footprint assessment.

The updated figure below shows that domain using the simulator's actual coordinate mapping. The corrected six-channel Zemax archive also provides concrete prescription differences to investigate. **The analytical model's overhanging R/K traces are a model discrepancy to resolve, not evidence that the optical design fails to fit.** No replacement rotations have been selected.

## What the corrected archive establishes

Source: `20260610B_ZS_6CH_Complete_Spectrograph.zar`, supplied on 2026-10-18. Its title is “ZShooter 6Ch. Spectrograph”; it contains 18 configurations, selecting minimum, middle and maximum orders for each channel. The notes identify June 2026 revision B, October 2026 pre-optics updates, and an H-band FSR update to the 65.6-degree blaze prescription.

The [extraction script](../validation/zshooter_xdisp/zemax_prescription.py) reads the archive with [zmxtools](https://github.com/tttom/zmxtools). It records the following values directly from the multi-configuration operands. Angles in this table retain their **Zemax coordinate-system meanings**.

| Channel | Orders | Echelle grooves/mm | CX grooves/mm | CX coordinate tilt X (deg) | Detector coordinate rotation Z (deg) | Image aperture X × Y (mm) |
|---|---:|---:|---:|---:|---:|---:|
| B | 85–118 | 50.15 | 2000 | +21.20 | +9.50 | 30.720 × 61.440 |
| G | 56–85 | 50.15 | 1100 | +16.75 | +7.00 | 30.720 × 61.440 |
| R | 37–56 | 50.15 | 625 | −14.10 | +6.00 | 30.720 × 61.440 |
| YJ | 57–80 | 24 | 350 | +10.90 | +5.00 | 36.864 × 36.864 |
| H | 42–51 | 24 | 300 | +13.70 | +6.10 | 36.864 × 36.864 |
| K | 31–38 | 24 | 270 | −16.80 | +6.50 | 36.864 × 36.864 |

These are surface 61's echelle parameters, surface 88's CX ruling, surface 85's coordinate tilt, surface 166's detector coordinate rotation, and surface 167's rectangular aperture. For a rectangular aperture, `APMN` and `APMX` are its X and Y **half-widths**, respectively; the table doubles them. See the [OpticStudio definitions](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Numeric_Functions.html). All values, including the camera centering tilts, are in [zemax_channels.csv](../validation/zshooter_xdisp/data/zemax_channels.csv).

The comparison with the current simulator is specific:

- **R and K order indices agree.** Their mismatch is not explained by an off-by-one error in the configured minimum or maximum order.
- **YJ is 350 grooves/mm**, confirming the restored ruling.
- **H is now 42–51**, whereas IRDB still has 41–51. This removes the reddest order; this archive does not add a higher-numbered blue order. Its maximum-order FSR ends near 1.497 microns and minimum-order FSR near 1.821 microns under the sampling reconstruction below.
- **The VIS echelle ruling is 50.15/mm**, versus 50.00/mm in IRDB and the paper's rounded prescription.
- **The NIR image aperture is 20% larger in each dimension** than the current 30.720 mm square model. A 2048-pixel detector spanning 36.864 mm would have 18 microns per pixel. The archive specifies the physical aperture; this pixel-pitch interpretation is conditional on 2048 pixels. The intended APD/H2RG configuration needs reconciling before changing detector dimensions.
- **The archive supplies optical angles directly.** The simulator instead estimates a CX angle that makes the nominal wavelength band fill the detector height. That artificial filling condition explains why its cross-dispersion footprints need not have the size selected by the optical designer. The Zemax coordinate tilts must be translated through the folded optical coordinate systems before using them as the simulator's incidence angles or detector rotations.

The mirror curvatures confirm 1500 and 1250 mm collimators. The existing analytical model already uses their 1.2 relay factor with the 225 mm camera, giving 270 mm effective echelle dispersion focal length. A separate paraxial calculation through the archive's actual camera surfaces and glass catalog gives 224.5, 224.3, 225.7, 225.1, 224.6 and 222.3 mm for B/G/R/YJ/H/K at their middle configuration's reference wavelength. These values support the nominal camera focal length; a mistaken 225 mm camera value does not explain the footprint discrepancy. The calculation includes the camera's refracting surfaces and thicknesses, but does not trace the complete folded spectrograph or its off-axis distortion.

## Wavelength sampling in the archive

The header declares three active fields and **11 active wavelengths**. Each configuration has 11 wavelength operands. They are equally spaced, and their spacing is consistent with **12 samples across an FSR, with the last sample absent**. The listed span is therefore approximately 10/11 of an FSR. This is a reproducible property of the supplied file; whether it was deliberate requires confirmation.

For example, extrapolating the existing spacing by one step gives:

| Channel/order | First listed wavelength (nm) | Last listed wavelength (nm) | One additional step would reach (nm) | Reconstructed full FSR (nm) |
|---|---:|---:|---:|---:|
| R / 37 | 966.757 | 990.835 | 993.243 | 26.486 |
| K / 31 | 2410.484 | 2482.331 | 2489.516 | 79.032 |

The inferred centers are 980 and 2450 nm, respectively, and the reconstructed widths agree with $\lambda_c/m$ to the rounding precision of the input wavelengths. Across all 18 configurations, the listed fraction is 90.89–90.92%. The extra endpoint in the table is **an extrapolation of the sampling pattern**, not a recovered Zemax ray intercept. The [original wavelength samples](../validation/zshooter_xdisp/data/zemax_wavelengths.csv) and [sampling calculation](../validation/zshooter_xdisp/data/zemax_sampling.csv) are retained separately.

This matters when comparing the analytical full-FSR figure with a Zemax plot using the file's listed wavelengths. Both must use the same wavelength domain and slit positions. It could account for part of the apparent R/K width disagreement, alongside the NIR aperture difference; it does not yet identify the complete optical cause. The active counts follow the `FTYP` fields used by the [Optiland Zemax reader](https://optiland.readthedocs.io/en/latest/_modules/optiland/fileio/zemax/reader/parser.html).

## Complete analytical FSR footprints

![Complete analytical FSRs and both slit ends](../validation/zshooter_xdisp/data/detector_footprints.png)

Each colored curve runs exactly between that configured order's FSR endpoints, with color increasing from blue to yellow in wavelength within each channel. Gray curves show both ends of the 10-arcsecond slit. Black rectangles are the configured detector boundaries. All orders are included, including portions outside the detector. The panels share the same X scale and have equal X/Y pixel scales.

The first row uses the original angle-estimation formula and original IRDB prescription. The second uses the corrected formula, restored YJ ruling and original rotations. The rotations remain B −6, G −6, R −2, YJ −4, H −3 and K −7 degrees. H still includes order 41 here because this is the **current configured simulation**, not a visualization of the newer Zemax prescription.

The plotted coordinates come from the same `SpectrographSetup` objects and mapping used to generate the trace tables, evaluated over their full FSRs. There is no separate plotting geometry or detector-edge extension. The mean tilt is the equal-weight mean of the end-to-end FSR tilt of each order. [footprints.csv](../validation/zshooter_xdisp/data/footprints.csv) gives channel summaries; [orders.csv](../validation/zshooter_xdisp/data/orders.csv) gives individual orders and bounds including both slit ends.

The unrotated echelle FSR widths are unchanged by the CX equation correction. The widest R and K FSRs in this model are 2168.0 and 2600.0 pixels: 32.520 and 39.000 mm at its configured 15-micron pixel size. These are internal model-axis spans, not widths measured along the rotated detector axes. The new plot exposes this existing disagreement instead of truncating the FSR at the detector boundary.

The placement calculation also needs attention. It currently centers the bounding box of traces sampled across an extended detector-X range. It does not center the full-FSR/slit envelope. Moreover, filling the nominal wavelength interval to the cross-dispersion boundaries leaves no explicit allowance for the outer orders' complete FSRs and slit ends. Fixing the scalar endpoint equation alone cannot correct those layout assumptions. A further rotation search would obscure these issues; the next geometry comparison must first reconcile the actual prescription and the wavelength/slit domain.

## Retained equation correction

The configured planar camera mapping is $y=f\tan(\beta-\beta_c)$. For wavelength endpoints at $y=\pm L/2$, subtracting the two grating equations gives

$$
h=\arctan\frac{L}{2f},\qquad
\cos\beta_c=\frac{\lambda_{\max}-\lambda_{\min}}{2d\sin h},\qquad
\alpha=\arcsin\left(\frac{\lambda_{\min}}d-\sin(\beta_c-h)\right).
$$

Here $d$ is groove spacing. Negative incidence angles are valid signed solutions. The old formula compressed the requested unrotated span by approximately 6.3%, 5.9%, 13.3%, 4.9%, 11.1% and 4.6% for B/G/R/YJ/H/K with the former prescription. The correction satisfies the stated planar mapping. Using a detector-filling estimate instead of the designer's angles is a separate modeling choice.

The two planar dispersion calculations and empirical rotations remain intentional approximations. No non-coplanar ray model or camera distortion model has been added here. To quantify their effect, the useful next reference is a set of actual image-plane ray intercepts at the same full-FSR endpoints, centers and slit ends, with the intended NIR detector configuration identified. A Zemax prescription extraction alone does not supply those intercepts.

## Verification and reproduction

The geometry inspection refactor was checked against all 127 existing generated trace tables: every numerical column is exactly unchanged. The FSR helper separately fixes a half-pixel offset by using the optical-axis wavelength, consistent with the existing central-wavelength and order-mask calculations. That small correction does not explain the R/K discrepancy.

All **22 targeted tests pass**, covering the endpoint equation, inverse mapping, complete FSR widths, slit extent, generated table coordinates, and missing placement. A known 100 mm thin lens checks the separate paraxial camera calculation in both propagation directions; missing glass data fails directly. The full six-channel baseline/current figure takes about four seconds to regenerate. No search or archive processing is added to normal observations.

All three repositories retain branch `fix/zshooter-xdisp-geometry`, based on ScopeSim `main` (`4830d68a`), IRDB `dev_master` (`d1415c4`), and workspace `main` (`6a616ea`). ScopeSim `acfa5a71` contains the angle-estimator correction and `aefbe60d` exposes the configured placement for full-FSR inspection. IRDB `cf88595` restores YJ's ruling. The rejected rotation trial `aefd527` was reverted by `cf9110a`. The obsolete coverage optimizer and its trial output tables have been removed from this report's active assessment; their history remains in Git.

To reproduce the analytical figure with the feature branches installed:

```sh
git -C ../ScopeSim show 4830d68a:scopesim/optics/echelle.py > /tmp/zshooter-baseline-echelle.py
git -C ../irdb show d1415c4:ZShooter_v2/traces/echelle_trace_parameters.dat > /tmp/zshooter-baseline-parameters.dat
conda run -n zssim python validation/zshooter_xdisp/geometry.py \
  --parameters ../irdb/ZShooter_v2/traces/echelle_trace_parameters.dat \
  --baseline-parameters /tmp/zshooter-baseline-parameters.dat \
  --baseline-echelle /tmp/zshooter-baseline-echelle.py \
  --output validation/zshooter_xdisp/data
```

The archive inspection additionally needs `zmxtools` 0.1.5. It was installed in a temporary directory for this investigation; it is not a simulator dependency. With that package available:

```sh
conda run -n zssim python validation/zshooter_xdisp/zemax_prescription.py \
  20260610B_ZS_6CH_Complete_Spectrograph.zar --output validation/zshooter_xdisp/data
```

The supplied archive remains untracked. Its SHA-256 is `774fbc539fc6e305ea000206a914ba2ea9f50f3e93912e910dd2ffd334e66070`. The exported CSVs are prescription values and explicitly labeled sampling calculations, not results from an OpticStudio execution.
