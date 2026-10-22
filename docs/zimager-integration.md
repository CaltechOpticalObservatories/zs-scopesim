# ZImager integration with the X-shooter stack

The three working checkouts are on `zimager_xshooter`. Each preserves its
`xshooter` and `zimager` histories with a merge commit. The main branches and
the existing `xshooter` branches have not moved. Nothing was pushed.

Start with [06_ZImager.ipynb](../notebooks/06_ZImager.ipynb). It runs a B1V, G1V,
M1V and flat-AB source through the three imaging arms, shows the independent-sky
subtracted images and retains the original aperture-photometry comparison.
The readouts are accumulated electrons, including the electronic bias, saved
as `.fits.gz` under `notebooks/06_ZImager/`.

## What changed

ScopeSim applies the imaging dichroics, arm optics, filter and QE to each
branch's spectrum before wavelength integration. Downstream emission stays in
optical order within that branch. The source is copied before branch filtering;
one arm cannot modify another arm's source or sky. The spectroscopic projection,
finite-cell integration, ADC ordering and existing PSF cache safeguards are
retained. No persistent branch cache was added.

IRDB now exposes `IMAGER`, `SPEC` and `SPEC_ALTDETECT` together. The imaging
configuration retains the three 4096 by 2304 CMOS detectors, 4.6 micrometre
pitch and approximately 0.0511 arcsec pixels. Its middle arm accepts Sloan
g, r or i. Dark current now enters the Poisson draw alongside source and sky.
The existing exposure-limit effect reports the expected charge per integration
and warns above the configured well without adjusting exposure settings or
clipping counts.

The notebook sets its epoch, seed, filter, seeing and exposures explicitly.
It reuses one prepared train for the sky after clearing the optical rate
images, with separate readout IDs for independent noise. The source and sky
each receive 300 seconds in every arm: 300 s x 1 in u and the middle arm,
150 s x 2 in z. The initial 300 s x 1 z run reached 7025 electrons per
integration against a 7000-electron well, motivating the visible z setting.
These are on-source times; the separate sky adds the same time again.
The displayed images subtract that sky; aperture photometry uses a local
annulus on the science frame, as in the original notebook.

| Arm | DIT x NDIT | Expected peak per integration | Usable well |
|---|---:|---:|---:|
| u | 300 s x 1 | 442 electrons | 7000 electrons |
| Middle, g | 300 s x 1 | 1579 electrons | 7000 electrons |
| z | 150 s x 2 | 3512 electrons | 7000 electrons |

## Verification

The [executed integration notebook](../validation/zimager/Integration.ipynb)
compares all 18 full detector-plane rate arrays against a baseline captured
before the merge. It covers ZShooter `SPEC` and `SPEC_ALTDETECT` at a 0.70 arcsec
slit, and X-shooter's faint and higher-resolution slit configurations. The
input is a flat AB20 point source with the configured optics, PSF, ADC and
optical thermal emission. Atmospheric acquisition is omitted to keep this
regression's input fixed. The CSV files beside that notebook record full-array
checksums, rates and timings.
**All 18 arrays match bit for bit.** The integrated and peak rates are unchanged.

The targeted tests cover wavelength-dependent branch colours, common and
downstream backgrounds, unchanged shared source data, missing branch IDs,
PSF cache invalidation, finite spectral/spatial cells, slit illumination,
readout exposure metadata, detector alternatives, extraction helpers and
notebook noise restoration. The new configured-imager tests additionally check
geometry, g -> r/i -> g changes against fresh trains, independent per-arm
exposures, stochastic photon/dark/read variance, readout replay and saturation warnings.
The selected suites pass: 210 ScopeSim cases and 98 workspace cases. Two ADC
test fixtures were updated to supply the FOV manager's spectroscopic mode.

The executed full-frame sky readouts also agree with their expected pixel
variance. Subtracting the configured bias and expected sky/dark charge, then
dividing by the predicted standard deviation, gives residual variances
1.00044, 0.99955 and 1.00025 in u, middle and z. The normalized residual means
have magnitude below 0.0003. This uses the actual detector-mapped rate images
and configured exposure, dark and read terms.

Imaging retains the effective broadband PSF approximation. The notebook's AB5
table is an empirical, background-dominated aperture estimate calibrated by
its flat-AB source; it does not include a source-shot-noise term in the
limiting-magnitude estimate. The original calculation and plot layout are
preserved.

## Runtime and memory

The final fresh-kernel ZImager run took 5.7 s to construct the train,
117.1 s for the source observation and 55.3 s for the sky observation.
Each readout took about 1.1 s. The two observations together therefore take
about three minutes, before plotting and writing files. Peak process memory,
including the full-frame noise check, was 8.9 GiB.

The spectroscopy regression observations took 47.1 s (`SPEC`), 48.6 s
(`SPEC_ALTDETECT`), 84.5 s (X-shooter faint) and 71.8 s (X-shooter higher
resolution). Their pre-merge times were 48.1, 52.9, 82.5 and 72.0 s. These
are single runs on the same workstation, not a claim of a speed improvement.

The 64 GiB workstation already had about 34 GiB of swap occupied before these
checks. The largest new validation kernel peaked at 9.4 GiB. New validation
kernels were closed afterwards; your existing sessions were preserved.
Restarting idle kernels whose arrays you no longer need would help more than
closing notebook tabs.

## Stack and review

The workspace manifest pins ScopeSim `b9ba36e6` and IRDB `e332b9e`. DRP remains
at `3fe83e09` and PyReduce at `9d9b8fe3`. Your pre-existing notebook reruns,
timing edits and report deletion were preserved byte for byte in the working
tree. Existing user kernels were left running.

The commits are local and ready for your review. No main-branch merge, PR,
upstream submission or CI change was made.
