"""Configured slit sampling and small isolated-line detector-phase experiments."""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.wcs import WCS
from synphot import SourceSpectrum, Empirical1D
from synphot.units import PHOTLAM
import scopesim as sim
from scopesim.optics.fov import FieldOfView3D
from zs_scopesim_tools.comparison import line_width
from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level


def isolated_profile(train, original, wavelength, displacement, quadrature=4):
    """Project one line through the real PSF/slit/trace into native detector pixels."""
    trace = train['trace_list_analytical'].spectral_traces[original.trace_id]
    trace._pixel_quadrature_order = quadrature
    trace._set_dispersion(trace.wave_min, trace.wave_max, pixsize=original.detector_header['CDELT1D'])
    pixel_wave = abs(trace.dlam_per_pix(wavelength))
    center = wavelength + displacement*pixel_wave
    sigma = center/(200000*np.sqrt(8*np.log(2)))
    wave = center + np.linspace(-8, 8, 161)*sigma
    spectrum = SourceSpectrum(Empirical1D, points=wave*u.um, lookup_table=np.exp(-.5*((wave-center)/sigma)**2)*PHOTLAM, fill_value=0)
    source = sim.Source(x=[0.], y=[0.], ref=[0], weight=[1.], spectra=[spectrum])
    low, high = (center-12*pixel_wave)*u.um, (center+12*pixel_wave)*u.um
    metadata = original.meta | {'wave_min': low, 'wave_max': high}
    probe = FieldOfView3D(original.header.copy(), (low, high), detector_header=original.detector_header, cmds=train.cmds, **metadata)
    probe.image_plane_id = original.image_plane_id
    probe.meta['spectral_padding'] = trace.spectral_padding(probe)
    psf = train['seeing_psf']
    probe.extract_from(source, spatial_header=psf.padded_header(probe, point_sources=True))
    psf.apply_to(probe)
    probe.cube = probe.hdu
    image = trace.map_spectra_to_focal_plane(probe)
    assert trace.dispersion_axis == 'x'
    # Sum the entire spatial cutout: the experiment holds extraction fixed.
    counts = image.data.sum(axis=0, dtype=np.float64)
    x, _ = WCS(image.header, key='D').all_pix2world(np.arange(len(counts)), np.zeros(len(counts)), 0)
    center_y = trace.xilam2y(0., center)
    wavelengths = trace.xy2lam(x, np.full(len(x), center_y))
    order = np.argsort(wavelengths)
    wavelengths, counts = wavelengths[order], counts[order]
    measured_center, rms, fwhm = line_width(wavelengths, counts)
    detector, = train.detector_managers[original.image_plane_id]
    native_x, _ = WCS(detector._native_header, key='D').all_world2pix(trace.xilam2x(0., center), center_y, 0)
    offsets = (wavelengths-center)/pixel_wave
    return center, pixel_wave, native_x % 1, counts.sum(), measured_center, rms, fwhm, offsets, counts


if __name__ == '__main__':
    started = perf_counter()
    configure_irdb_path()
    set_scopesim_log_level('error')
    output = Path(__file__).parent/'data/resolution_sampling'
    output.mkdir(exist_ok=True)
    geometry_rows, phase_rows, profile_rows, checks = [], [], [], []
    cases = [('natural .70', .70, False), ('natural .33', .33, False), ('AO .33', .33, True)]
    for name, slit, ao in cases:
        commands = sim.UserCommands(use_instrument='ZShooter_v2', properties={'!OBS.mjdobs': 60310.25, '!OBS.seeing': .65, '!OBS.airmass': 1.1, '!INST.vis_curr_slit': slit, '!INST.nir_curr_slit': slit})
        for document in commands.yaml_dicts:
            if 'effects' in document:
                document['effects'] = [effect for effect in document['effects'] if effect['name'] not in ('atmo_transmission', 'continuum_emission', 'airglow_and_interline_continuum')]
        train = sim.OpticalTrain(commands)
        psf = train['seeing_psf']
        psf.meta['enable_ao'] = ao
        traces = train['trace_list_analytical']
        if name == 'natural .70':
            for channel, spectrograph in traces._spectrographs.items():
                for order in spectrograph.orders:
                    trace_id = f'{channel}_{order}'
                    trace = traces.spectral_traces[trace_id]
                    wave = spectrograph.central_wave(order) + np.linspace(-.5, .5, 101)*spectrograph.fsr(order)
                    # Use the actual trace and native detector WCS, including placement.
                    detector, = train.detector_managers[trace.meta['image_plane_id']]
                    header = detector._native_header
                    pixel_wcs = WCS(header, key='D')
                    displacement = trace.slit_dispersion(wave.to_value(u.um))*.70/2
                    xi = np.zeros(len(wave))
                    xlo, _ = pixel_wcs.all_world2pix(trace.xilam2x(xi, wave.to_value(u.um)-displacement), trace.xilam2y(xi, wave.to_value(u.um)-displacement), 0)
                    xhi, _ = pixel_wcs.all_world2pix(trace.xilam2x(xi, wave.to_value(u.um)+displacement), trace.xilam2y(xi, wave.to_value(u.um)+displacement), 0)
                    x, y = pixel_wcs.all_world2pix(trace.xilam2x(xi, wave.to_value(u.um)), trace.xilam2y(xi, wave.to_value(u.um)), 0)
                    covered = (x >= -.5) & (x <= header['NAXIS1']-.5) & (y >= -.5) & (y <= header['NAXIS2']-.5)
                    width = abs(xhi-xlo)
                    threshold = 2*.70/width
                    narrow_shift = trace.slit_dispersion(wave.to_value(u.um))*threshold/2
                    narrow_low, _ = pixel_wcs.all_world2pix(trace.xilam2x(xi, wave.to_value(u.um)-narrow_shift), trace.xilam2y(xi, wave.to_value(u.um)-narrow_shift), 0)
                    narrow_high, _ = pixel_wcs.all_world2pix(trace.xilam2x(xi, wave.to_value(u.um)+narrow_shift), trace.xilam2y(xi, wave.to_value(u.um)+narrow_shift), 0)
                    np.testing.assert_allclose(abs(narrow_high-narrow_low), 2, rtol=2e-6)
                    for wavelength, pixel_width, narrow_slit, on_detector in zip(wave.to_value(u.um), width, threshold, covered):
                        geometry_rows.append((channel, order, wavelength, pixel_width, narrow_slit, on_detector))
        for channel, spectrograph in traces._spectrographs.items():
            order = spectrograph.orders[len(spectrograph.orders)//2]
            trace_id = f'{channel}_{order}'
            original, = [fov for fov in train.fov_manager.fovs if fov.trace_id == trace_id]
            center = spectrograph.central_wave(order).to_value(u.um)
            fwhm_arcsec = psf.fwhm(np.array([center])*u.um)[0].to_value(u.arcsec)
            for displacement in np.linspace(0, 1, 17):
                result = isolated_profile(train, original, center, displacement)
                actual, pixel_wave, phase, flux, centroid, rms, fwhm, offsets, counts = result
                phase_rows.append((name, channel, order, actual, displacement, phase, slit, fwhm_arcsec, psf.alpha, flux, actual/fwhm, fwhm/pixel_wave, rms/pixel_wave, (centroid-actual)/pixel_wave))
                if displacement in (0., .25, .5, .75):
                    normalized = counts/counts.max()
                    profile_rows.extend((name, channel, displacement, offset, count) for offset, count in zip(offsets, normalized))
                if displacement == .5:
                    refined = isolated_profile(train, original, center, displacement, quadrature=16)
                    checks.append((name, channel, refined[3]/flux-1, refined[6]/fwhm-1, (refined[4]-centroid)/pixel_wave))
            print(name, channel, 'complete', flush=True)
        del train
    geometry = pd.DataFrame(geometry_rows, columns=['channel', 'order', 'wavelength_um', 'slit_070_width_pixel', 'two_pixel_slit_arcsec', 'on_detector'])
    phases = pd.DataFrame(phase_rows, columns=['configuration', 'channel', 'order', 'wavelength_um', 'displacement_pixel', 'native_pixel_phase', 'slit_arcsec', 'psf_fwhm_arcsec', 'moffat_beta', 'flux', 'R_fwhm', 'fwhm_pixel', 'rms_pixel', 'centroid_error_pixel'])
    profiles = pd.DataFrame(profile_rows, columns=['configuration', 'channel', 'displacement_pixel', 'offset_pixel', 'normalized_counts'])
    convergence = pd.DataFrame(checks, columns=['configuration', 'channel', 'flux_fraction', 'fwhm_fraction', 'centroid_pixel'])
    for table, filename in [(geometry, 'slit_sampling.csv'), (phases, 'pixel_phase.csv'), (profiles, 'profiles.csv'), (convergence, 'quadrature_check.csv')]:
        table.to_csv(output/filename, index=False)
    assert convergence.flux_fraction.abs().max() < .01
    assert convergence.fwhm_fraction.abs().max() < .02
    assert convergence.centroid_pixel.abs().max() < .01
    # Moving an otherwise identical line across a pixel must preserve photons.
    phase_flux = phases.groupby(['configuration', 'channel']).flux.agg(['min', 'max'])
    assert (phase_flux['max']/phase_flux['min']-1).max() < .01
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout='constrained')
    for index, (channel, group) in enumerate(geometry[geometry.on_detector].groupby('channel', sort=False)):
        for order_index, (_, order_data) in enumerate(group.groupby('order')):
            label = channel.upper() if order_index == 0 else None
            axes[0].plot(order_data.wavelength_um, order_data.slit_070_width_pixel, color=f'C{index}', lw=.8, label=label)
            axes[1].plot(order_data.wavelength_um, order_data.two_pixel_slit_arcsec, color=f'C{index}', lw=.8)
    axes[0].set(ylabel='Projected .70 arcsec full slit [pixels]', title='Actual configured detector x projection; covered primary FSRs')
    axes[0].legend(ncol=6)
    axes[1].axhline(.33, color='.3', ls='--', lw=1, label='.33 arcsec slit')
    axes[1].set(ylabel='Slit giving two detector x pixels [arcsec]', xlabel=r'Wavelength [$\mu$m]')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/'slit_sampling.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for ax, (channel, group) in zip(axes.flat, phases.groupby('channel', sort=False)):
        for name, selected in group.groupby('configuration', sort=False):
            ax.plot(selected.displacement_pixel, selected.R_fwhm/1000, '.-', ms=3, label=name)
        ax.set(title=f'{channel.upper()}, order {group.order.iloc[0]}, {group.wavelength_um.iloc[0]:.3f} '+r'$\mu$m', xlabel='Line displacement [native pixel]', ylabel=r'$R_{\mathrm{FWHM}}$ / 1000')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Same PSF, slit and optical configuration; only line position relative to pixels changes')
    fig.savefig(output/'pixel_phase.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout='constrained')
    for row, channel in enumerate(('b', 'k')):
        for column, (name, _, _) in enumerate(cases):
            ax = axes[row, column]
            for displacement, group in profiles[(profiles.channel == channel) & (profiles.configuration == name)].groupby('displacement_pixel'):
                ax.plot(group.offset_pixel, group.normalized_counts, '.-', ms=4, lw=.8, label=f'{displacement:g} pixel')
            ax.axhline(.5, color='.5', lw=.6)
            ax.set(title=f'{channel.upper()}: {name}', xlim=(-4, 4), xlabel='Offset from incident line [native pixels]', ylabel='Peak-normalized integrated counts')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Native detector samples joined only to show the half-height width estimator')
    fig.savefig(output/'profiles.png', dpi=170)
    print(phases.groupby(['configuration', 'channel']).agg({'R_fwhm':['min', 'max'], 'fwhm_pixel':['min', 'max']}).round(3).to_string())
    print('Quadrature changes:', convergence.iloc[:, 2:].abs().max().to_dict())
    print(f'Total elapsed: {perf_counter()-started:.1f} s')
