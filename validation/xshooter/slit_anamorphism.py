"""Separate projected slit width, linear dispersion and slit-limited resolution."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import scopesim as sim
from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level

configure_irdb_path()
set_scopesim_log_level('error')
commands = sim.UserCommands(use_instrument='ZShooter_v2', properties={'!OBS.mjdobs': 60310.25})
for document in commands.yaml_dicts:
    if 'effects' in document:
        document['effects'] = [effect for effect in document['effects'] if effect['name'] not in ('atmo_transmission', 'continuum_emission', 'airglow_and_interline_continuum')]
train = sim.OpticalTrain(commands)
rows = []
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True, layout='constrained')
fraction = np.linspace(-.5, .5, 201)
for channel, spectrograph in train['trace_list_analytical']._spectrographs.items():
    grating = spectrograph.grating
    for order in spectrograph.orders:
        center = spectrograph.central_wave(order)
        wave = center + fraction*spectrograph.fsr(order)
        half_angle = (.35*u.arcsec*spectrograph.entrance_slit_scale/spectrograph.collimator1_focal_length).value*u.rad
        beta_low = np.arcsin(order*wave/grating.d-np.sin(grating.alpha+half_angle))
        beta_high = np.arcsin(order*wave/grating.d-np.sin(grating.alpha-half_angle))
        finite_width = abs(spectrograph.x_pixel_for_beta(beta_high)-spectrograph.x_pixel_for_beta(beta_low))
        width = spectrograph.projected_slit_width(wave, order, .7*u.arcsec).to_value(u.pixel)
        step = center*1e-7
        dispersion = (spectrograph.wavelength_to_x_pixel(wave+step, order)-spectrograph.wavelength_to_x_pixel(wave-step, order))/(2*step)
        resolution = (wave*dispersion/finite_width).to_value(u.dimensionless_unscaled)
        expected = (wave/abs(spectrograph.slit_dispersion(wave, order)*.7*u.arcsec)).to_value(u.dimensionless_unscaled)
        np.testing.assert_allclose(width, finite_width.value, rtol=2e-7)
        np.testing.assert_allclose(resolution, expected, rtol=2e-7)
        rows.append((channel, order, width.min(), width.max(), resolution.min(), resolution.max(), np.ptp(width)/width[100], np.ptp(resolution)/resolution[100]))
        if order == spectrograph.orders[len(spectrograph.orders)//2]:
            for ax, values in zip(axes, (width, dispersion.value, resolution)):
                ax.plot(fraction, 100*(values/values[100]-1), label=f'{channel.upper()}, order {order}')
for ax, title in zip(axes, ('Projected slit width', 'Detector pixels per wavelength', 'Geometric slit-limited resolving power')):
    ax.set(title=title, xlabel='Position within primary FSR', ylabel='Change from blaze center [%]')
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
output = Path(__file__).parent/'data/reconciliation'
output.mkdir(exist_ok=True)
fig.savefig(output/'slit_anamorphism.png', dpi=180)
table = pd.DataFrame(rows, columns=['channel', 'order', 'slit_width_min_pixel', 'slit_width_max_pixel', 'R_min', 'R_max', 'slit_width_fractional_span', 'R_fractional_span'])
table.to_csv(output/'slit_anamorphism.csv', index=False)
print(table.groupby('channel').agg({'slit_width_min_pixel':'min', 'slit_width_max_pixel':'max', 'R_min':'min', 'R_max':'max', 'slit_width_fractional_span':'max', 'R_fractional_span':'max'}).to_string())
