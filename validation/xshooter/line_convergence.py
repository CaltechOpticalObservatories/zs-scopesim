"""Refine finite slit, spectral and detector integration using configured traces."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import ndtr
import scopesim as sim
from scopesim.effects.psfs.analytical import moffat_cell_integrals
from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level


def line_probe(trace, detector_header, center, slit, fwhm, eta_offset, refinement=1, phase=0., quadrature=4, cube_dtype=np.float64):
    """An isolated narrow line with cell-integrated effective Moffat illumination."""
    trace._pixel_quadrature_order = quadrature
    trace._set_dispersion(trace.wave_min, trace.wave_max, pixsize=detector_header['CDELT1D'])
    pixel_wave = abs(trace.dlam_per_pix(center))
    center += phase * pixel_wave
    spectral_step = abs(trace.slit_dispersion(center))*min(slit, fwhm) / (20*refinement)
    half_range = max(center / 2000, 12 * pixel_wave)
    wave = center + np.arange(-int(half_range/spectral_step), int(half_range/spectral_step) + 1) * spectral_step
    sigma = pixel_wave / 10
    line = np.diff(ndtr((np.r_[wave - spectral_step/2, wave[-1] + spectral_step/2] - center) / sigma)) / spectral_step
    eta_rows = int(np.ceil(slit / (fwhm / (10 * refinement))))
    deta = slit / eta_rows
    eta = (np.arange(eta_rows) + .5) * deta - slit/2
    dxi = fwhm / (6 * refinement)
    xi = np.arange(-int(2/dxi), int(2/dxi) + 1) * dxi
    beta = 4.765
    gamma = fwhm / (2 * np.sqrt(2**(1/beta) - 1))
    illumination = moffat_cell_integrals(xi[None, :], (eta - eta_offset)[:, None], dxi, deta, np.array([gamma]), beta) / (dxi * deta)
    cube = (line[:, None, None] * illumination[None, :, :]).astype(cube_dtype)
    header = fits.Header({'CTYPE1': 'LINEAR', 'CTYPE2': 'LINEAR', 'CTYPE3': 'WAVE', 'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec', 'CUNIT3': 'um', 'CDELT1': dxi, 'CDELT2': deta, 'CDELT3': spectral_step, 'CRPIX1': 1., 'CRPIX2': 1., 'CRPIX3': 1., 'CRVAL1': xi[0], 'CRVAL2': eta[0], 'CRVAL3': wave[0]})
    metadata = {'wave_min': wave[0]*u.um, 'wave_max': wave[-1]*u.um, 'xi_min': xi[0]*u.arcsec, 'xi_max': xi[-1]*u.arcsec, 'eta_min': -slit/2*u.arcsec, 'eta_max': slit/2*u.arcsec}
    metadata['effective_psf_fwhm'] = fwhm*u.arcsec
    fov = SimpleNamespace(cube=fits.ImageHDU(cube, header), header=header, detector_header=detector_header, meta=metadata, trace_id=trace.trace_id)
    image = trace.map_spectra_to_focal_plane(fov)
    yy, xx = np.indices(image.data.shape)
    x, y = WCS(image.header, key='D').all_pix2world(xx, yy, 0)
    mapped_wave = trace.xy2lam(x, y)
    photons = image.data.sum(dtype=np.float64)
    centroid = center + (image.data * (mapped_wave-center)).sum(dtype=np.float64) / photons
    sigma = np.sqrt((image.data * (mapped_wave-centroid)**2).sum(dtype=np.float64) / photons)
    return photons, centroid, sigma, pixel_wave


if __name__ == '__main__':
    configure_irdb_path()
    set_scopesim_log_level('error')
    rows = []
    for instrument, trace_name, channels in [('ZShooter_v2', 'trace_list_analytical', ['b', 'g', 'r', 'yj', 'h', 'k']), ('XSHOOTER', 'spectral_traces', ['UVB', 'VIS', 'NIR'])]:
        commands = sim.UserCommands(use_instrument=instrument, properties={'!OBS.mjdobs': 60310.25})
        # Source/sky acquisition is irrelevant to this monochromatic geometry check.
        for document in commands.yaml_dicts:
            if 'effects' in document:
                document['effects'] = [effect for effect in document['effects'] if effect['name'] not in ('atmo_transmission', 'continuum_emission', 'airglow_and_interline_continuum', 'skycalc_atmosphere')]
        train = sim.OpticalTrain(commands)
        for channel in channels:
            candidates = [trace for name, trace in train[trace_name].spectral_traces.items() if name.startswith(channel + '_')]
            trace = candidates[len(candidates)//2]
            detector_header = train.image_planes[trace.meta['image_plane_id']].header
            center = (trace.wave_min + trace.wave_max)/2
            for fwhm in (.65, .12):
                for phase in (0., .25, .5, .75):
                    for refinement in (1, 2, 4):
                        flux, centroid, sigma, pixel_wave = line_probe(trace, detector_header, center, .33, fwhm, .0825, refinement, phase)
                        rows.append((instrument, channel, trace.trace_id, fwhm, phase, refinement, 4, flux, centroid, sigma, pixel_wave))
                    flux, centroid, sigma, pixel_wave = line_probe(trace, detector_header, center, .33, fwhm, .0825, 4, phase, 16)
                    rows.append((instrument, channel, trace.trace_id, fwhm, phase, 4, 16, flux, centroid, sigma, pixel_wave))
            print(f'{instrument} {channel} complete', flush=True)
    result = pd.DataFrame(rows, columns=['instrument', 'channel', 'trace', 'fwhm_arcsec', 'pixel_phase', 'refinement', 'pixel_quadrature', 'flux', 'centroid_um', 'sigma_um', 'pixel_um'])
    output = Path(__file__).parent / 'data/line_convergence.csv'
    result.to_csv(output, index=False)
    reference = result[result.pixel_quadrature == 16].set_index(['instrument', 'channel', 'fwhm_arcsec', 'pixel_phase'])
    baseline = result[(result.refinement == 1) & (result.pixel_quadrature == 4)].set_index(reference.index.names)
    differences = pd.DataFrame({'flux_fraction': baseline.flux/reference.flux - 1, 'width_fraction': baseline.sigma_um/reference.sigma_um - 1, 'centroid_pixel': (baseline.centroid_um-reference.centroid_um)/reference.pixel_um})
    differences.to_csv(output.with_name('line_convergence_errors.csv'))
    print(differences.abs().groupby(['instrument', 'channel']).max().to_string())
    assert differences.flux_fraction.abs().max() < .01
    assert differences.width_fraction.abs().max() < .02
    assert differences.centroid_pixel.abs().max() < .01
