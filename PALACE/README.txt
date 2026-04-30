Paranal Airglow Line and Continuum Emission (PALACE) model

Python/Cython code:
Author: Stefan Noll (st@noll-x.de)
Version: 1.0 (10/2024)
Licence: GNU GPLv3 (see LICENSE file and https://www.gnu.org/licenses/)

Model data (data/palace_*.fits):
Authors: Stefan Noll (st@noll-x.de), Carsten Schmidt, Patrick Hannawald,
         Wolfgang Kausch, and Stefan Kimeswenger
Version: 1.0 (10/2024)
Licence: CC-BY-4.0 (see https://creativecommons.org/licenses/by/4.0/)

Files included in PALACE/:
README.txt
LICENSE
setup_cython.py
setup_python.py
src/palace/__init__.py
src/palace/palace.py
src/palace/config/palace_default.par
src/palace/data/palace_cont.fits
src/palace/data/palace_lines.fits
src/palace/data/palace_var.fits
src/scripts/palace_run.py

Instructions for use:

PALACE calculates nighttime airglow spectra in the wavelength regime from 0.3
to 2.5 µm for Cerro Paranal in Chile. Line and continuum emissions from OH,
O2, HO2, FeO, Na, K, O, N, and H are involved. The emission strength depends
on the given zenith angle, local time (solar mean time at Cerro Paranal),
month, and solar radio flux at 10.7 cm. Optionally, atmospheric absorption and
scattering of airglow photons can be considered. The different input
parameters are described in the default parameter file (palace_default.par).

For running the code, python3 packages such as setuptools, numpy, matplotlib,
and astropy as well as cython should be available. Successful tests were
performed for Python versions from 3.8 to 3.13. Up to 3.9, this also required
the installation of importlib_resources. On Linux systems, the PALACE Python
package can be installed by typing

cp setup_python.py setup.py
pip3 install .

in the PALACE folder if pip3 is present. Then, the PALACE functions and an
executable are available independent of the location. For this purpose, the
necessary files will be stored in ~/.local/lib/python3.X/site-packages
(with X to be replaced by the version) and ~/.local/bin by default. The latter
should be included in the path variable (echo $PATH). In order to benefit from
a speed-up of the calculations by about an order of magnitude by means of
Cython-based functions, the module palace.py should be compiled. For Python
versions up to 3.11, this can be achieved by means of

cp setup_cython.py setup.py
pip3 install .

, which requires a C compiler (gcc). The installation can be removed by

pip3 uninstall palace

.

For running the code with the default values in an arbitrary folder, the
simple command

palace_run.py

is sufficient. The parameter values in src/palace/config/palace_default.par
can be changed with the given limitations. In this case, the modified file
should be saved with a different name outside the src folder to protect the
default version. The modified parameter file (e.g. palace.par) can be used by

palace_run.py parfile=palace.par

. It is also possible to provide other parameter values via the command line.
For example,

palace_run.py mbin=6 srf=120. isatm=False lammin=0.7 lammax=2.2

calculates a spectrum for June, for a solar radio flux of 120 sfu, without
atmospheric absorption and scattering, and a wavelength range from 0.7 to
2.2 µm. The other parameters are taken from palace_default.par. In a Python
shell, the same can be achieved by

python3
>>> from palace import palace
>>> lst = ['mbin=6', 'srf=120.', 'isatm=False', 'lammin=0.7', 'lammax=2.2']
>>> parlist = palace.parlist(lst)
>>> spec = palace.model(**parlist)
>>> palace.output(spec, **parlist)

. parlist is a dictionary and spec is an astropy table. If all parameters are
provided via a parameter file (e.g. palace.par), the parameter list can also
be created by

>>> parlist = palace.parlist([], parfile='palace.par')

. If also a plot on the screen is desired, either 'showplot=True' needs to be
added to the parameter list or 

>>> palace.plotspec(spec, **parlist)

has to be called. Finally, the output spectrum can directly be written by

>>> palace.writespec(spec, **parlist)

.

PALACE produces a spectrum in micrometres (µm) and rayleighs per nanometre
(R/nm) of the combined airglow emission or of a specific atmospheric species
for the given parameter values. The spectrum also includes a 1-sigma
uncertainty estimate related to the variability not covered by the
climatological model. By default, the spectrum is written into a FITS table
with the path and name given by the parameters outdir and outname. Missing
directories are created. If the parameter specsuffix is not set to 'fits', an
ASCII file is written. Moreover, the parameters used are written into a
parameter file (suffix 'par') in the output directory, which is similar to 
palace_default.par but without comments.

PALACE relies on the data in the FITS tables palace_lines.fits,
palace_cont.fits, and palace_var.fits, which provide the list of mean line
intensities, spectra of unresolved emission (pseudo-continua), and the
climatological variability data. If the code is run in a folder different from
the PALACE directory, it is important to change the relative path in the
parameter datadir. PALACE is based on measurements and theoretical data. The
measurements are related to the X-shooter and UVES echelle spectrographs of
the Very Large Telescope, where large fractions of the archive were analysed
to obtain mean values and variations. The theoretical data comprise properties
such as wavelengths, energy levels, Einstein-A coefficients, and recombination
coefficients.

The description of the complex analysis for the build-up of PALACE and an
evaluation of the quality of the resulting model should be published in a
separate paper with the title "PALACE v1.0: Paranal Airglow Line And Continuum
Emission model" that was submitted by Noll et al. to Geoscientific Model
Development (GMD). The data sets related to the article are part of the same
Zenodo release as for this code in v1.0.
