import numpy as np
import pytest
import astropy.units as u
import os

from zs_scopesim_tools.comparison import boxcar_sum, gaussian_bin_matrix, overlap_bin_matrix, limiting_abmag, line_width, select_spectral_samples, save_comparison_case, load_comparison_case


def test_optional_case_save_preserves_spectra_and_excludes_native_images(tmp_path):
    configuration = {'instrument': 'ZShooter_v2', 'mode': 'SPEC_ALTDETECT4K', 'seeing': .65}
    result = {'configuration': configuration, 'wave': np.array([1., 1.1])*u.um, 'source_rate': np.array([2., 3.]), 'native_images': [np.ones((4, 4))]}
    filename = tmp_path/'case.pkl'
    save_comparison_case(filename, result)
    restored = load_comparison_case(filename, configuration, [])
    np.testing.assert_equal(restored['wave'], result['wave'])
    assert restored['wave'].unit == u.um
    np.testing.assert_equal(restored['source_rate'], result['source_rate'])
    assert restored['native_images'] is None
    assert result['native_images'][0].shape == (4, 4)
    save_comparison_case(filename, result, include_images=True)
    restored = load_comparison_case(filename, configuration, [])
    np.testing.assert_equal(restored['native_images'][0], result['native_images'][0])


def test_explicit_case_load_fails_for_missing_or_obviously_stale_inputs(tmp_path):
    configuration = {'mode': 'SPEC_ALTDETECT4K', 'seeing': .65}
    filename = tmp_path/'case.pkl'
    with pytest.raises(FileNotFoundError):
        load_comparison_case(filename, configuration, [])
    save_comparison_case(filename, {'configuration': configuration, 'native_images': None})
    with pytest.raises(ValueError, match='settings differ'):
        load_comparison_case(filename, configuration | {'seeing': .8}, [])
    model = tmp_path/'detector.yaml'
    model.write_text('pixel_size: 0.010\n')
    newer = filename.stat().st_mtime_ns+1_000_000_000
    os.utime(model, ns=(newer, newer))
    with pytest.raises(ValueError, match='model inputs are newer'):
        load_comparison_case(filename, configuration, [tmp_path])


def test_boxcar_extracts_disjoint_native_pixels_on_either_axis():
    image = np.arange(48.).reshape(6, 8)
    pixels = np.array([1, 3, 5])
    centers = np.array([2., 2.5, 3.])
    widths = np.array([1, 2, 3])
    expected = np.array([image[2, 1], image[2:4, 3].sum(), image[2:5, 5].sum()])
    np.testing.assert_equal(boxcar_sum(image, 'x', pixels, centers, widths), expected)
    np.testing.assert_equal(boxcar_sum(image.T, 'y', pixels, centers, widths), expected)


def test_boxcar_preserves_faint_counts_beside_a_bright_single_precision_pixel():
    image = np.array([[1e8], [1.], [1.]], dtype=np.float32)
    result = boxcar_sum(image, 'x', np.array([0]), np.array([1.]), 3)
    assert result[0] == 100000002.


def test_smoothing_conserves_counts_and_propagates_covariance():
    wave = np.exp(np.linspace(-.01, .01, 101)) * u.um
    edges = np.exp(np.linspace(-.03, .03, 61)) * u.um
    operator = gaussian_bin_matrix(wave, edges, .002)
    np.testing.assert_allclose(np.array(operator.sum(axis=0)).ravel(), 1, atol=3.e-9)
    variance = np.linspace(1, 3, len(wave))
    covariance = (operator.multiply(variance) @ operator.T).toarray()
    weights = np.zeros(len(edges) - 1)
    weights[27:33] = 1
    direct = np.sum((operator.T @ weights)**2 * variance)
    propagated = weights @ covariance @ weights
    assert propagated == pytest.approx(direct)
    assert propagated > np.sum(weights**2 * np.diag(covariance))


def test_limiting_magnitude_includes_source_shot_noise():
    signal = 100.
    background_variance = 400.
    mag = limiting_abmag(signal, signal, background_variance, reference_mag=20., snr=5.)
    amplitude = 10**((20 - mag)/2.5)
    assert amplitude * signal / np.sqrt(amplitude * signal + background_variance) == pytest.approx(5.)
    assert limiting_abmag(100., 100., 0.) == pytest.approx(20 - 2.5*np.log10(.25))


def test_avalanche_selection_keeps_physical_signal_and_scales_only_photon_variance():
    signal = np.array([10., 10., 10.])
    background_variance = np.array([2., 2., 10000.])
    selected, magnitude = select_spectral_samples(signal, background_variance, excess_noise_factor=1.2)
    assert set(selected) == {0, 1}
    amplitude = 10**((20 - magnitude)/2.5)
    snr = amplitude*signal[selected].sum()/np.sqrt(1.2*amplitude*signal[selected].sum() + background_variance[selected].sum())
    assert snr == pytest.approx(5.)


def test_native_bin_overlap_preserves_counts_and_shared_pixel_covariance():
    operator = overlap_bin_matrix(np.array([1., 2., 3.])*u.um, np.array([1., 1.5, 2.5, 3.])*u.um)
    counts = np.array([12., 20.])
    np.testing.assert_allclose(operator @ counts, [6., 16., 10.])
    covariance = (operator.multiply(counts) @ operator.T).toarray()
    assert covariance.sum() == pytest.approx(counts.sum())
    assert covariance[0, 1] == pytest.approx(3.)
    assert covariance[1, 2] == pytest.approx(5.)


def test_independent_equal_time_sky_and_gaussian_rebin_variance():
    rng = np.random.default_rng(74319)
    source, sky, dark, ron, exposures = 3., 17., .4, 2.5, 2
    draws = 50000
    on = rng.poisson(exposures*(source+sky+dark), (draws, 4)) + rng.normal(0, np.sqrt(exposures)*ron, (draws, 4))
    off = rng.poisson(exposures*(sky+dark), (draws, 4)) + rng.normal(0, np.sqrt(exposures)*ron, (draws, 4))
    weights = np.array([.1, .5, 1., .4])
    measured = (on-off) @ weights
    expected_variance = np.sum(weights**2)*exposures*(source+2*(sky+dark+ron**2))
    assert measured.mean() == pytest.approx(exposures*source*weights.sum(), abs=.15)
    assert measured.var() == pytest.approx(expected_variance, rel=.02)


def test_measured_gaussian_width_and_units():
    wavelength = np.linspace(.995, 1.005, 1001) * u.um
    sigma = .0004 * u.um
    counts = np.exp(-.5 * ((wavelength - 1*u.um) / sigma).value**2)
    center, measured_sigma, fwhm = line_width(wavelength, counts)
    assert center.to_value(u.um) == pytest.approx(1.)
    assert measured_sigma.to_value(u.um) == pytest.approx(sigma.value)
    assert fwhm.to_value(u.um) == pytest.approx(2*np.sqrt(2*np.log(2))*sigma.value, rel=1.e-4)


def test_float32_line_centroid_retains_subpixel_precision():
    pixel = 2e-5*u.um
    wavelength = 2*u.um+np.linspace(-8, 8, 1601)*pixel
    expected = 2*u.um+.37*pixel
    counts = (1e8*np.exp(-.5*((wavelength-expected)/pixel).value**2)).astype(np.float32)
    center, _, _ = line_width(wavelength, counts)
    assert abs(((center-expected)/pixel).value) < 1e-5


def test_cannot_smooth_to_a_sharper_resolution():
    with pytest.raises(ValueError, match='target resolution'):
        gaussian_bin_matrix(np.array([1.])*u.um, np.array([.99, 1.01])*u.um, 0.)


def test_spectral_selection_rejects_bright_sky_without_spatial_reweighting():
    signal = np.array([10., 10., 10., 10.])
    background = np.array([1., 10000., 2., 3.])
    selected, magnitude = select_spectral_samples(signal, background)
    assert set(selected) == {0, 2, 3}
    assert magnitude > limiting_abmag(signal.sum(), signal.sum(), background.sum())
