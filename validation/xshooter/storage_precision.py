"""Isolate float32 cube storage from grid, operator and reduction precision."""
from pathlib import Path
import numpy as np
import pandas as pd
import scopesim as sim
from line_convergence import line_probe
from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level


if __name__ == '__main__':
    configure_irdb_path()
    set_scopesim_log_level('error')
    rows = []
    for instrument, trace_name, channels in [('ZShooter_v2', 'trace_list_analytical', ['b', 'g', 'r', 'yj', 'h', 'k']), ('XSHOOTER', 'spectral_traces', ['UVB', 'VIS', 'NIR'])]:
        commands = sim.UserCommands(use_instrument=instrument, properties={'!OBS.mjdobs': 60310.25})
        for document in commands.yaml_dicts:
            if 'effects' in document:
                document['effects'] = [effect for effect in document['effects'] if effect['name'] not in ('atmo_transmission', 'continuum_emission', 'airglow_and_interline_continuum', 'skycalc_atmosphere')]
        train = sim.OpticalTrain(commands)
        for channel in channels:
            candidates = [trace for name, trace in train[trace_name].spectral_traces.items() if name.startswith(channel + '_')]
            trace = candidates[len(candidates)//2]
            detector_header = train.image_planes[trace.meta['image_plane_id']].header
            center = (trace.wave_min + trace.wave_max)/2
            cases = [(.70, .65), (.33, .65), (.33, .12)] if instrument == 'ZShooter_v2' else [(.9, .65)]
            for slit, fwhm in cases:
                for phase in (0., .37):
                    reference = line_probe(trace, detector_header, center, slit, fwhm, slit/4, phase=phase, cube_dtype=np.float64)
                    stored = line_probe(trace, detector_header, center, slit, fwhm, slit/4, phase=phase, cube_dtype=np.float32)
                    flux64, center64, sigma64, pixel = reference
                    flux32, center32, sigma32, _ = stored
                    rows.append((instrument, channel, slit, fwhm, phase, flux32/flux64-1, sigma32/sigma64-1, (center32-center64)/pixel))
            print(instrument, channel, 'complete', flush=True)
    result = pd.DataFrame(rows, columns=['instrument', 'channel', 'slit_arcsec', 'fwhm_arcsec', 'pixel_phase', 'flux_fraction', 'width_fraction', 'centroid_pixel'])
    result.to_csv(Path(__file__).parent / 'data/storage_precision.csv', index=False)
    errors = result[['flux_fraction', 'width_fraction', 'centroid_pixel']].abs().max()
    print(errors.to_string())
    # These isolate storage rounding; they are deliberately much tighter than
    # the 1% flux, 2% width and .01-pixel grid-convergence requirements.
    assert errors.flux_fraction < 1e-6
    assert errors.width_fraction < 1e-6
    assert errors.centroid_pixel < 1e-5
