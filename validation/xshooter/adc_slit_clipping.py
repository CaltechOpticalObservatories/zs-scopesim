"""Compare corrected and legacy ADC/slit ordering at the GREEN 5000-Angstrom feature.

Run with the installed simulation stack; ``configure_irdb_path`` locates IRDB.
This renders one configured slit FOV, without observations or detector noise.
The atmosphere/background effects are omitted because the diagnostic compares
spatial operations on the same source. Optical geometry and PSF remain those
of the kilonova notebook: 0.7-arcsec slit/seeing, airmass 1.3, AO off.

The corrected renderer is checked against the configured PSF's cell integrals
at the same integer displacement before slit clipping. The continuous residual
is shown separately: integer quantization remains an intentional approximation.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scopesim as sim
from synphot import SourceSpectrum
from synphot.models import ConstFlux1D
from zs_scopesim_tools.helpers import configure_irdb_path, warning_prevent_sync_alt_ra_dec

configure_irdb_path()
sim.utils.set_console_log_level('ERROR')
commands = sim.UserCommands(use_instrument='ZShooter_v2', set_modes=['SPEC'])
properties = {
    '!OBS.airmass': 1.3, '!OBS.seeing': .7, '!OBS.brightness': 'dark', '!OBS.mjdobs': 60310.25,
    '!INST.vis_curr_slit': .7, '!INST.nir_curr_slit': .7, '!ATMO.ao_enabled': False,
    '!INST.include_diffuse_emissivity': False, '!SIM.spectral.spectral_bin_width': 1e-5,
}
for key, value in properties.items():
    commands[key] = value
warning_prevent_sync_alt_ra_dec(commands)
for document in commands.yaml_dicts:
    if 'effects' in document:
        document['effects'] = [effect for effect in document['effects'] if effect['name'] not in ('atmo_transmission', 'continuum_emission', 'airglow_and_interline_continuum')]
train = sim.OpticalTrain(commands)
fov = next(fov for fov in train.fov_manager.fovs if fov.trace_id == 'g_73')
psf, adc = train['seeing_psf'], train['adc_residuals']
trace = train['trace_list_analytical'].spectral_traces[fov.trace_id]
fov.meta['spectral_padding'] = trace.spectral_padding(fov)
source = sim.Source(x=[0], y=[0], ref=[0], weight=[1], spectra=[SourceSpectrum(ConstFlux1D, amplitude=20*u.ABmag)])
legacy = deepcopy(fov)
support = adc._padded_header(psf.padded_header(fov, point_sources=True), adc.get_shifts(fov))
for view in (legacy, fov):
    view.extract_from(source, spatial_header=support)
psf.apply_to(legacy)
before = legacy.hdu.data.sum(axis=(1, 2))
adc.apply_to(legacy)
retained = legacy.hdu.data.sum(axis=(1, 2))/before
psf._apply_convolution(fov, spatial_shifts=[adc])
corrected = fov.hdu.data.sum(axis=(1, 2))/before
wave = WCS(fov.hdu.header).spectral.pixel_to_world(np.arange(fov.hdu.data.shape[0])).to_value(u.AA)
dx, dy = adc.get_shifts(fov)
cell_x, cell_y = fov.hdu.header['CDELT1']*u.deg, fov.hdu.header['CDELT2']*u.deg
# Display the two rounding operations in ShiftFoV3D.apply_to/roll_shift.
# The retained photons below come from actually applying that effect.
roll_x = np.round(np.round((dx/cell_x).decompose().value, 1)).astype(int)
roll_y = np.round(np.round((dy/cell_y).decompose().value, 1)).astype(int)

selected = (wave > 4970) & (wave < 5025)
centered = psf.point_source_image(fov, 0., 0.).sum()
preslit_integer = np.empty(selected.sum())
for xstep, ystep in np.unique(np.column_stack((roll_x[selected], roll_y[selected])), axis=0):
    same_shift = (roll_x[selected] == xstep) & (roll_y[selected] == ystep)
    image = psf.point_source_image(fov, (xstep*cell_x).to_value(u.arcsec), (ystep*cell_y).to_value(u.arcsec))
    preslit_integer[same_shift] = image.sum()/centered
np.testing.assert_allclose(corrected[selected], preslit_integer, rtol=1e-6)

# Evaluate the unquantized displacement at explicit wavelengths before,
# within and after the trough; do not interpolate a replacement PSF model.
rows = []
for reference_wave in (4980., 4995., 5010.):
    index = np.argmin(abs(wave-reference_wave))
    pre_actual = psf.point_source_image(fov, dx[index].to_value(u.arcsec), dy[index].to_value(u.arcsec)).sum()/centered
    pre_integer = psf.point_source_image(fov, (roll_x[index]*cell_x).to_value(u.arcsec), (roll_y[index]*cell_y).to_value(u.arcsec)).sum()/centered
    rows.append((wave[index], dx[index].to_value(u.arcsec), dy[index].to_value(u.arcsec), roll_x[index], roll_y[index], retained[index], corrected[index], pre_integer, pre_actual))
table = pd.DataFrame(rows, columns=['wavelength_AA', 'adc_x_arcsec', 'adc_y_arcsec', 'roll_x_cell', 'roll_y_cell', 'postslit_roll_fraction', 'corrected_fraction', 'preslit_integer_fraction', 'preslit_actual_fraction'])

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
axes[0].plot(wave[selected], dy[selected].to_value(u.arcsec), label='Configured ADC residual')
axes[0].plot(wave[selected], roll_y[selected]*cell_y.to_value(u.arcsec), label='Applied integer row shift')
axes[0].set(ylabel='Across-slit shift [arcsec]')
axes[1].plot(wave[selected], retained[selected], label='Legacy ADC roll after slit clipping')
axes[1].plot(wave[selected], corrected[selected], label='Corrected renderer: shift incoming illumination')
axes[1].plot(wave[selected], preslit_integer, '--', label='PSF integral: same integer displacement')
axes[1].plot(table.wavelength_AA, table.preslit_actual_fraction, 'o', ms=4, color='black', label='PSF integral: continuous ADC residual')
axes[1].set(ylabel='Source flux / centered source flux', xlabel='Wavelength [Angstrom]', ylim=(.89, 1.01))
for ax in axes:
    ax.legend(fontsize=9)
output = Path(__file__).parent/'data/kilonova_reduction'
output.mkdir(parents=True, exist_ok=True)
fig.savefig(output/'adc_slit_clipping.png', dpi=150)
table.to_csv(output/'adc_slit_clipping.csv', index=False)
print(f'Configured g_73 cube: {fov.hdu.data.shape}; cell widths {cell_x.to(u.arcsec)}, {cell_y.to(u.arcsec)}')
print(table.to_string(index=False))
