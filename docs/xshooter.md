# X-shooter detector simulations

`notebooks/01_Kilonovae_XSHOOTER.ipynb` is the detector-only companion to the ZShooter kilonova notebook. It uses the same source construction and produces three ScopeSim HDULists, one per arm. The `xshooter` branches contain the feature commits; `env/zshooter-stack.toml` pins the ScopeSim and IRDB revisions used by this notebook.

This is a first-pass comparison model. **Five-percent accuracy across all wavelengths, slit widths, backgrounds, and observing conditions is not established.** Optical throughput is calibrated against an archived standard star, while the NIR response and stray light still have larger uncertainties. The numerical trace-fit accuracy is not an absolute instrument accuracy claim.

The [shared-physics assessment](xshooter-validation.md) distinguishes calibration from independent validation, records the literature/report review and current ESO QC comparison, and identifies shared numerical limitations. The separate `fidelity-candidates` branches contain experiments for ZShooter and the common engine; the pinned notebook stack does not select them. Agreement with X-shooter's fitted throughput or resolving power alone does not validate ZShooter predictions.

## Using the instrument

```python
import scopesim as sim
from zs_scopesim_tools.helpers import configure_irdb_path

configure_irdb_path()
cmds = sim.UserCommands(use_instrument="XSHOOTER", set_modes=["SPEC"], properties={
    "!OBS.airmass": 1.3, "!OBS.seeing": 0.7,
    "!INST.uvb_curr_slit": "1.0", "!INST.vis_curr_slit": "0.9", "!INST.nir_curr_slit": "0.9",
    "!OBS.dit_uvb": 300., "!OBS.ndit_uvb": 1,
    "!OBS.dit_vis": 300., "!OBS.ndit_vis": 1,
    "!OBS.dit_nir": 300., "!OBS.ndit_nir": 1,
    "!DET.binx_uvb": 1, "!DET.biny_uvb": 2,
    "!DET.binx_vis": 1, "!DET.biny_vis": 2,
})
train = sim.OpticalTrain(cmds)
train.observe(source)  # The same ScopeSim Source used for ZShooter.
readouts = train.readout()
```

Construct a new optical train after changing the optical configuration or atmosphere. The package composes `Paranal_spectroscopy.yaml`, `VLT_UT3_Cassegrain.yaml`, and the XSHOOTER instrument/detector files. The telescope has the UT3 Cassegrain M1/M2 path. A later Keck comparison should replace that telescope component explicitly; adding a second telescope YAML would apply both optical paths. The empirical optical response is conditional on the adopted UT3 mirror curve.

The companion notebook ends at detector images. It does not invoke `zsdrp`, extract spectra, resample instruments onto a common grid, or change the original notebook.

The executed example writes `kilonova_1.5d_uvb.fits`, `kilonova_1.5d_vis.fits`, `kilonova_1.5d_nir.fits`, and `native_frames.png` into `notebooks/xshooter_kilonova/`. All 43 orders mapped successfully in the full-band run. The three arrays contain finite values and remain below the stated detector saturation thresholds for this example.

## Supported science configuration

| Property | UVB | VIS | NIR |
|---|---|---|---|
| Nominal wavelength range | 0.30–0.56 um | 0.55–1.02 um | 1.02–2.48 um |
| Orders in the calibrated format | 13–24 (12) | 16–30 (15) | 11–26 (16) |
| Native science array, FITS x by y | 2048 by 3000 | 2048 by 4000 | 2048 by 1100 |
| Physical pixel pitch | 15 um | 15 um | 18 um |
| Slit length | 11 arcsec | 11 arcsec | 11 arcsec |
| Slit widths, arcsec | 0.5, 0.8, 1.0, 1.3, 1.6, 5.0 | 0.4, 0.7, 0.9, 1.2, 1.5, 5.0 | 0.4, 0.6, 0.9, 1.2, 5.0; 0.6JH, 0.9JH |
| Default slit | 1.0 | 0.9 | 0.9 |
| Reference resolving power at default slit | 5400 | 8900 | 5600 |
| On-chip binning, x by y | 1 by 1, 1 by 2, 2 by 2 | 1 by 1, 1 by 2, 2 by 2 | 1 by 1 |

These are the standard raw science regions, not the complete physical silicon dimensions. No optional windows are exposed. The optical prescan/overscan is omitted. The NIR orientation and 1100-row readout follow an archived standard exposure. Calibration tables describe actual curved orders, including their wavelength overlap; the nominal band limits above are not a rectangular order layout.

For the CCDs, FITS x is the spatial direction and y the dispersion direction, matching the manual's spatial-by-spectral binning convention. Thus `(1, 2)` bins two dispersion pixels.

The manual's overview still says 14 VIS orders, but its detailed format table and the archived calibration contain 15. The latter determine the simulation.

## Detector counts and noise

CCD settings `!DET.readout_mode_uvb` and `!DET.readout_mode_vis` accept `slow` (100 kHz, high gain) and `fast` (400 kHz, low gain). Gains in electrons/ADU are 0.63/1.83 for UVB and 0.64/1.64 for VIS; corresponding read noises are 2.4/4.6 and 3.4/6.6 electrons. NIR gain is 2.29 electrons/ADU.

Photon rate is integrated over DIT and NDIT. Dark charge is added per native pixel, then CCD charge is summed into on-chip bins, Poisson noise is drawn, and one amplifier read noise is applied per output pixel per exposure. NIR read noise is interpolated from ESO's DIT table over 1–1800 s. NIR sums are divided by NDIT before gain conversion. Thus a NIR output has the mean signal of one DIT and variance reduced by NDIT. CCD outputs sum exposures. For a native rate `s`, dark rate `d`, read noise `r`, and gain `g`, the NIR output has mean `(s+d)*DIT/g` and variance `((s+d)*DIT+r**2)/(NDIT*g**2)`.

The optical dark-current default is the manual's **upper bound** of 2 electrons/hour/pixel, not a precision measurement. NIR uses 0.02 electrons/s/pixel. These are explicit configurable detector properties.

Outputs are floating-point, bias-subtracted ADU expectations with noise. They are not byte-for-byte ESO raw products. Negative noise samples are retained. `ARM`, `DIT`, `NDIT`, `EXPTIME`, `GAIN`, `BINX`, `BINY`, `SLIT`, and `COMBINE` describe the count convention. `EXPTIME` is accumulated DIT times NDIT. The detector `D` WCS uses physical focal-plane millimetres and follows binning; the generic ScopeSim sky WCS is not an echelle wavelength solution. The coordinate-inspection helper currently rejects binned readouts rather than returning unbinned pixel positions under a binned label.

Use unsaturated exposures: CCD values approaching 65000 ADU and NIR values approaching the approximately 42000 ADU threshold are outside this linear model. Saturation, threshold-limited ramp extrapolation, persistence, individual bad pixels, fringing realizations, readout overheads, and exact ESO raw-container conventions are not simulated.

## Optical and background model

The offline preparation script evaluates ESO's calibrated prism/echelle physical model and fits the resulting wavelength/slit/focal-plane tables. Simulation uses those tables directly. A compiler and the attributed ESO C kernel are needed only to regenerate them. Forward and inverse mappings are checked on a separate sampling grid; all 43 orders meet 0.03 native-pixel RMS and 0.1-pixel maximum interpolation error.

The cross-slit wavelength derivative comes from the same optical model. It is independent of the numerical sky pixel size and on-chip binning. The along-slit center value is used; its variation over the 11 arcsec slit is below 0.04% in the bundled calibration. The finite slit illumination is integrated before an optical Gaussian line-spread function is applied. Optical resolving powers of 17000, 43000, and 46000 were fitted to the manual's slit-resolution table using Gaussian fits to slit- and pixel-convolved lines. They are optical blur terms, not the final slit resolving powers. Default-slit estimates agree within about 3%; the widest tabulated optical slits differ by about 6–7%. Point-source resolution also depends on seeing and centering.

Native detector rasterization uses ScopeSim's pixel-center quadrature with the local coordinate Jacobian. The slit-resolution estimates are therefore approximate; intra-pixel response and subpixel-phase effects on narrow lines are not calibrated.

UVB and VIS use order response measured from the first 2024-01-01 EG21 nod exposure after subtracting the other nod. The calculation divides by the pipeline reference flux, exposure, collecting area, atmospheric extinction, adopted mirror reflectivity, and detector QE. Robust 32-pixel continuum bins suppress unflattened fixed-pattern structure and cosmic rays; interpolation bridges excluded molecular bands and holds endpoint throughput outside the sampled interval. These gaps and endpoint continuations are not independent throughput calibrations. The second nod is retained as a repeatability check, not evidence for an independent absolute flux scale.

NIR uses the ESO ETC optics/QE curves, its empirical correction, and a scalar echelle blaze approximation. This has not passed the 5% absolute-throughput target. The JH blocking curve is digitized from manual Figure 13 near its 2.06 um cutoff, with an explicit 0.945 passband continuation below 2 um. Fine filter fringes and the localized K-band light leak are omitted.

Paranal transmission and sky radiance come from ESO SkyCalc, with airmass and PWV explicitly connected to the observation. `!OBS.seeing` is zenith seeing at 500 nm. The Moffat PSF uses alpha 4.765 and scales its FWHM with wavelength to the power -0.2 and airmass to the power 0.6. UVB/VIS ADCs are assumed ideal within their documented airmass range (up to 2); NIR refraction relative to 1.31 um is active. The fixed default MJD records the geometry-calibration epoch; change observing coordinates/time deliberately when changing slit orientation.

NIR warm stray light is added after dispersion and QE, separately from sky, telescope emission, and dark current. The manual supplies four measured rates normalized to a one-arcsec slit. The package interpolates a cross-dispersion profile through their modeled detector positions, holds endpoint rates beyond those positions, and scales by the selected slit width. Its dependence on detector position outside those measurements and on ambient temperature is not calibrated. This is a material limitation for faint NIR comparisons.

## Validation and reproducibility

The IRDB package contains `calibration/trace_residuals.ecsv`, `calibration/standard_repeatability.ecsv`, the measured continuum bins, pinned model parameters, and the scripts that prepare the runtime FITS files. The tests exercise native array sizes, supported readouts, gains, charge conservation, repeated readouts, WCS, noise statistics, spectral mapping, and calibration bounds. See the package README for input identifiers and regeneration commands.

Representative narrow-band continuum tests at 0.438, 0.711, and 1.634 um compare the simulated electron rate with a direct integral of source photon surface brightness, slit area, collecting area, mirror transmission, order response, and QE. Agreement is within 1.4%. Monochromatic trace-mapping tests conserve photons within 0.1% at numerical sky spacings of 0.1 and 0.05 arcsec and reproduce the variance from input width, slit illumination, and optical blur. These establish numerical normalization, not absolute instrument calibration.

Centered point-source tests at the same wavelengths agree within 0.4% with direct integrals of the configured Moffat profile through the slit, including the wavelength and airmass dependence of seeing.

The broader regression review found 26 failures and 20 setup errors in the original ScopeSim fork and exactly the same failing cases on the feature and candidate branches. The [assessment](xshooter-validation.md#regression-results) records these inherited failures, the clean upstream comparison, and the real-instrument observations. The fork suite is not fully passing. The separate zs-scopesim lamp-amplitude and zero-redshift failures also predate this instrument work.

The optical second-nod RMS is below 5% in 26 of 27 orders. UVB order 13 reaches about 16% when all sampled continuum bins are included, chiefly near its low-response edges. VIS order 17 has only four accepted continuum bins because of absorption; its interpolation is weakly constrained. Neither these internal checks nor the trace interpolation test certifies 5% end-to-end performance. Absolute throughput, NIR background, telluric bands, and the stated slit-resolution outliers remain the principal limitations.

An independent ESO QC sample returned 27 observations per arm from August through September 2026. The monitored optical efficiency medians are 1.6–2.8% below the January 2024 calibration epoch, supporting retention of that optical response for a first pass. All sample entries have `photometric=N`; scatter, limited wavelength coverage, and the five-arcsec calibration slit prevent a general accuracy claim. NIR medians are 8–11% higher than the 2024 observations, an epoch comparison rather than a measurement of this model's NIR error. No response curves were refitted to these data. The [saved measurements and interpretation](xshooter-validation.md#independent-efficiency-check) separate this check from the internal calibration tests.

## Primary references

- [ESO X-shooter documentation](https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc.html) and [User Manual v10, 2024-08-19](https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_P114v1.pdf): detector properties, current slits, JH transmission, background, resolution, and detailed order format.
- [Vernet et al. 2011](https://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/xshooter_Vernet2011.pdf): optical design and instrument characterization.
- [Pipeline manual 3.8.12](https://ftp.eso.org/pub/dfs/pipelines/instruments/xshooter/xshoo-pipeline-manual-3.8.12.pdf) and [ESO pipeline source kit](https://ftp.eso.org/pub/dfs/pipelines/instruments/xshooter/xshoo-kit-3.8.12-3.tar.gz): calibrated forward optical model and raw-coordinate conventions.
- [ESO ETC calibration database](https://www.eso.org/observing/etc/bin/simu/calibdisp?INS.NAME=XSHOOTER+INS.MODE%3Dnone): efficiency, QE, empirical corrections, and NIR ramp noise.
- [ESO SkyCalc](https://www.eso.org/observing/etc/skycalc/): Paranal sky radiance and transmission.
