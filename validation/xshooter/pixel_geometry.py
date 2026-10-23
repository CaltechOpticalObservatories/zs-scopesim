"""Bound the within-pixel affine approximation against each configured trace."""
from pathlib import Path
import numpy as np
import pandas as pd
import scopesim as sim
from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level

configure_irdb_path()
set_scopesim_log_level('error')
rows = []
for instrument, effect_name in [('ZShooter_v2', 'trace_list_analytical'), ('XSHOOTER', 'spectral_traces')]:
    commands = sim.UserCommands(use_instrument=instrument, properties={'!OBS.mjdobs': 60310.25})
    for document in commands.yaml_dicts:
        if 'effects' in document:
            document['effects'] = [effect for effect in document['effects'] if effect['name'] not in ('atmo_transmission', 'continuum_emission', 'airglow_and_interline_continuum', 'skycalc_atmosphere')]
    train = sim.OpticalTrain(commands)
    for name, trace in train[effect_name].spectral_traces.items():
        wave, xi = np.meshgrid(np.linspace(trace.wave_min, trace.wave_max, 101), [-4., 0., 4.])
        x, y = trace.xilam2x(xi, wave), trace.xilam2y(xi, wave)
        pitch = trace.meta['pixel_size']
        ax, ay = trace.xy2xi.gradient()
        bx, by = trace.xy2lam.gradient()
        ax, ay, bx, by = ax(x, y), ay(x, y), bx(x, y), by(x, y)
        spatial_step = pitch*np.hypot(ax, ay)
        spectral_step = pitch*np.hypot(bx, by)
        spatial_error, spectral_error = 0., 0.
        for dx in (-pitch/2, pitch/2):
            for dy in (-pitch/2, pitch/2):
                approximate_xi = trace.xy2xi(x, y)+ax*dx+ay*dy
                approximate_wave = trace.xy2lam(x, y)+bx*dx+by*dy
                spatial_error = max(spatial_error, np.max(abs(trace.xy2xi(x+dx, y+dy)-approximate_xi)/spatial_step))
                spectral_error = max(spectral_error, np.max(abs(trace.xy2lam(x+dx, y+dy)-approximate_wave)/spectral_step))
        rows.append((instrument, name, spatial_error, spectral_error))
result = pd.DataFrame(rows, columns=['instrument', 'trace', 'spatial_error_pixel', 'spectral_error_pixel'])
result.to_csv(Path(__file__).parent/'data/pixel_linearization.csv', index=False)
print(result.groupby('instrument')[['spatial_error_pixel', 'spectral_error_pixel']].max().to_string())
assert result.spatial_error_pixel.max() < .001
assert result.spectral_error_pixel.max() < .001
