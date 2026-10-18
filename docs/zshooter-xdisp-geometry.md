# ZShooter: three analytical geometry options

The figure contains exactly these three options, from top to bottom. No option has been selected for adoption.

| Row | Rulings and order numbers | CX angles | Detector rotations |
|---|---|---|---|
| **1. Corrected CX estimation** | Zemax | Corrected angle-estimation equation | Zemax |
| **2. Explicit Zemax CX angles** | Zemax | Explicit Zemax angles in the existing analytical model | Zemax |
| **3. Main** | Main | Main's original angle-estimation equation | Main |

![The three requested analytical geometry options](../validation/zshooter_xdisp/data/detector_footprints.png)

Colored curves show every order's complete FSR at slit center; gray curves show both ends of the 10-arcsecond slit. Black rectangles are the configured detectors. All three rows use the same wavelength colors and axis limits within each channel, equal X/Y pixel scales, and a common X range. Traces continue through their full FSRs, including portions outside the detector. Mean tilt weights orders equally.

**Main's detector-placement calculation is used in all three rows.** The later full-FSR recentering change is excluded. Camera/effective dispersion focal lengths remain 225/270 mm; pixels remain 15 micrometers; detectors remain 2048 x 4096 in VIS and 2048 x 2048 in NIR. Each case also retains main's FSR endpoint convention.

## Inputs actually used

The archive is `20260610B_ZS_6CH_Complete_Spectrograph.zar`. Options 1 and 2 share the following rulings, orders and rotations:

| Channel | Orders, options 1/2 | Echelle grooves/mm, options 1/2 | CX grooves/mm, options 1/2 | Detector rotation, options 1/2 (deg) | Detector rotation, main (deg) |
|---|---:|---:|---:|---:|---:|
| B | 85-118 | 50.15 | 2000 | -9.50 | -6.00 |
| G | 56-85 | 50.15 | 1100 | -7.00 | -6.00 |
| R | 37-56 | 50.15 | 625 | -6.00 | -2.00 |
| YJ | 57-80 | 24 | 350 | -5.00 | -4.00 |
| H | 42-51 | 24 | 300 | -6.10 | -3.00 |
| K | 31-38 | 24 | 270 | -6.50 | -7.00 |

Main differs in three additional entries: its VIS echelle ruling is **50.00/mm**, its YJ CX ruling is **325/mm**, and its H orders are **41-51**. The other rulings and order limits match the table.

The actual CX incidence / camera-reference angles, in degrees, are:

| Channel | Option 1: corrected estimation | Option 2: explicit Zemax | Option 3: main |
|---|---:|---:|---:|
| B | 16.41 / 27.49 | 21.20 / 21.20 | 24.74 / 18.91 |
| G | 8.68 / 26.56 | 16.75 / 16.75 | 16.51 / 18.20 |
| R | -7.58 / 40.33 | 14.10 / 14.10 | 1.85 / 28.72 |
| YJ | 10.78 / 12.48 | 10.90 / 10.90 | 4.15 / 17.58 |
| H | -6.49 / 37.55 | 13.70 / 13.70 | 2.45 / 26.94 |
| K | 10.93 / 24.25 | 16.80 / 16.80 | 17.85 / 17.06 |

Option 2 uses the existing explicit-angle convention: the supplied angle sets both CX incidence and camera-reference angle. Its value is `-sign(camera_space) * CX_tilt` from the archive. Detector point rotations in options 1/2 are the negatives of the archive's detector-coordinate rotations. These are the conversions used for this analytical comparison.

Full-precision values are in [settings.csv](../validation/zshooter_xdisp/data/settings.csv); [orders.csv](../validation/zshooter_xdisp/data/orders.csv) contains the full-FSR/slit bounds, and [footprints.csv](../validation/zshooter_xdisp/data/footprints.csv) contains channel summaries. The [archive extraction](../validation/zshooter_xdisp/data/zemax_channels.csv) retains its original coordinate angles.

## Reproduce and verify

The comparison pins ScopeSim `main` to `4830d68a` and IRDB's corresponding `dev_master` to `d1415c4`. It reads those committed inputs directly from Git, independently of the sibling repositories' current checkouts. The inspection methods from `aefbe60d` expose the existing trace mapping; the original main FSR equation and placement are retained. Only options 1/2 use the corrected angle-estimation source, and option 2 bypasses estimation through its explicit angles.

```sh
conda run -n zssim python validation/zshooter_xdisp/geometry.py --scopesim ../ScopeSim --irdb ../irdb --output validation/zshooter_xdisp/data
conda run -n zssim python -m pytest tests/test_xdisp_geometry.py tests/test_zemax_prescription.py -q
```

All **29 targeted tests pass**. Option 3 reproduces every column of all **127 unmodified-main trace tables exactly**. Options 1 and 2 each plot all **126 configured orders**. The checks cover corrected-estimator endpoints, explicit-angle bypass, shared inputs, complete FSR/slit curves and common plot scales. The figure was visually inspected and takes about **8 seconds** to generate without running observations.

The test run emits ScopeSim's existing `fov_grid` deprecation warnings and Astropy polynomial-fit conditioning warnings. The figure uses the direct analytical mapping rather than the fitted polynomials.
