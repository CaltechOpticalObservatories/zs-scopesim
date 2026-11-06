"""Line projection used by notebooks/03_Resolution_and_Sampling.ipynb."""
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from synphot import SourceSpectrum, Empirical1D
from synphot.units import PHOTLAM
import scopesim as sim
from scopesim.optics.fov import FieldOfView3D
from zs_scopesim_tools.comparison import line_width


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
