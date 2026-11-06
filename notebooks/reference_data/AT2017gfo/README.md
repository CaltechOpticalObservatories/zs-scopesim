# AT2017gfo X-shooter reference spectra

These two unchanged numerical tables were downloaded on 2026-10-20 from the [ENGRAVE data release](https://www.engrave-eso.org/AT2017gfo-Data-Release/):

- [MJD 57983.969, release phase +1.43 d](https://sid.erda.dk/share_redirect/df1fMhon6Z/flux_corrected_smoothed_joined_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d.dat)
- [MJD 57990.000, release phase +7.40 d](https://sid.erda.dk/share_redirect/df1fMhon6Z/flux_corrected_smoothed_joined_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57990.000_Phase%2B7.40d.dat)

Cite both original papers when using these data: [Pian et al. (2017), Nature 551, 67](https://doi.org/10.1038/nature24298) and [Smartt et al. (2017), Nature 551, 75](https://doi.org/10.1038/nature24303). ENGRAVE supplied the subsequent uniform reduction by Selsing and photometric calibration, joining and smoothing by Gillanders. The [release notes](https://sid.erda.dk/share_redirect/df1fMhon6Z/README.txt) describe the corrections, including the revised first-epoch UVB calibration.

The notebook plots column 1 (wavelength in Angstrom) against column 2 (telluric-corrected flux in erg/s/cm^2/Angstrom). These observed-frame spectra retain Galactic extinction. They are photometrically calibrated and already median-smoothed over 51 samples, with additional smoothing at arm joins. No further smoothing or normalization is applied. The fourth column is ambiguously headed `variance/error`; it is not used to infer S/N. Initial zero-filled flux samples are masked for plotting.

[Pian et al., Extended Data Table 2](https://arts.units.it/bitstream/11368/2921596/4/2921596_nature24298-PostPrint.pdf) gives the following observing conditions. Its rounded phases differ from the release filenames:

| Release phase | Paper phase | UVB/VIS/NIR slits (arcsec) | Exposure (s) | Airmass range | Seeing (arcsec) |
| --- | --- | --- | --- | --- | --- |
| 1.43 d | 1.5 d | 1.0 / 0.9 / 0.9 | 2400 | 1.45–1.72 | 0.61 |
| 7.40 d | 7.5 d | 1.0 / 0.9 / 0.9 | 3600 | 1.66–2.44 | 0.53 |

The plots compare these observed fluxes with the simulated 1.5-, 7.5- and 20-day scenes. They do not adjust the observations to the simulations' exposure time, epoch, atmospheric conditions or detector binning. No 20-day observed spectrum is supplied here.
