"""Contrast the supplied report's sampling stress cases with resolved inputs.

Usage: representative_sampling.py path/to/yashvi_demos
Select the engine with PYTHONPATH and the instrument checkout with IRDB.
This uses the report's exact Moffat quadrature and source construction.
"""
import argparse
import sys
import numpy as np
import astropy.units as u
from synphot import SourceSpectrum, Empirical1D
from synphot.units import PHOTLAM

parser = argparse.ArgumentParser()
parser.add_argument('demos')
args = parser.parse_args()
sys.path.insert(0, args.demos)
from _common import train, fov_of, fresh_fov, effect, point_source, moffat_fraction, in_slit_light

tr = train(.70, **{'!OBS.mjdobs': 60310.25})
fov = fresh_fov(fov_of(tr, 'g_66'), .5700, .5702)
fov.extract_from(point_source(2., 0.))
fov.view()
psf = effect(tr, 'seeing_psf')
psf.apply_to(fov)
fwhm = psf.fwhm(.5701 * u.um).to_value(u.arcsec)
exact = moffat_fraction(fwhm, psf.alpha, 2., 0., -5., 5., -.35, .35)
print(f'Ordinary seeing: FWHM={fwhm:.6f} arcsec; in-slit/exact={in_slit_light(fov)[0] / exact:.6f}')

step = tr.cmds['!SIM.spectral.spectral_bin_width'] * 1e4
for resolving_power in (20000, 200000):
    ratios = []
    for phase in np.linspace(0, 1, 7, endpoint=False):
        centre = 5600. + phase * step
        sigma = centre / resolving_power / 2.3548
        nodes = np.r_[2000., np.arange(centre - 8 * sigma, centre + 8 * sigma, sigma / 20), 30000.]
        flux = 1. + 100. * np.exp(-.5 * ((nodes - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        flux[0] = flux[-1] = 1.
        spectrum = SourceSpectrum(Empirical1D, points=nodes * u.AA, lookup_table=flux * PHOTLAM)
        exact_line = np.trapezoid(flux - 1., nodes)
        prepared = tr.prepare_source(point_source(2., 0., spectrum)).fields[0].spectra[0]
        grid = prepared.waveset.to_value(u.AA)
        selected = abs(grid - centre) < 3.
        line_flux = ((prepared(grid[selected]).value - 1.) * step).sum()
        ratios.append(line_flux / exact_line)
    print(f'Line R={resolving_power}; wavelength step={step:.3f} A; prepared/exact range={min(ratios):.6f}..{max(ratios):.6f}')
