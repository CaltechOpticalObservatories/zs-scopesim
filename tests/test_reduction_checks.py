import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pytest
from astropy.io import fits
from scipy import sparse
from pyreduce.spectra import Spectra, Spectrum

from zs_scopesim_tools.reduction_checks import flat_ab_extraction_bins, flat_ab_band_integrals


def write_flat_spectrum(path, wave, factor=1.):
    flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
    flux = (20*u.ABmag).to_value(flux_unit, equivalencies=u.spectral_density(wave*u.AA))
    sigma = 1e-18
    covariance = sparse.diags([np.full(len(wave)-1, .3*sigma**2), np.full(len(wave), sigma**2), np.full(len(wave)-1, .3*sigma**2)], [-1, 0, 1], format='csr')
    spectrum = Spectrum(m=None, spec=factor*flux, wave=wave, sig=np.full(len(wave), sigma), covariance=covariance)
    header = fits.Header({'BUNIT': 'erg s-1 cm-2 Angstrom-1', 'COVUNIT': 'erg2 s-2 cm-4 Angstrom-2'})
    Spectra(header=header, data=[spectrum]).save(path)
    return spectrum


def test_width_weighted_flux_and_correlated_uncertainty(tmp_path):
    path = tmp_path/'flat.fits'
    spectrum = write_flat_spectrum(path, np.arange(4000., 4021.), factor=1.2)
    table, figure = flat_ab_extraction_bins({'BLUE': path}, intervals={'BLUE': [(4000*u.AA, 4020*u.AA)]}, max_gap={'BLUE': 1.1*u.AA}, magnitude=20, bin_R=1000)
    np.testing.assert_allclose(table['ratio'], 1.2)
    first = table[0]
    edges = np.arange(3999.5, 4021.)
    overlap = np.maximum(0, np.minimum(edges[1:], first['wave_high'].value)-np.maximum(edges[:-1], first['wave_low'].value))
    weights = overlap/overlap.sum()
    expected_variance = weights @ (spectrum.covariance @ weights)
    np.testing.assert_allclose(first['uncertainty'].value**2, expected_variance)
    assert expected_variance > np.sum(weights**2*spectrum.sig**2)
    np.testing.assert_allclose(table['snr'], table['flux']/table['uncertainty'])
    np.testing.assert_allclose(table['covered_width'].sum(), 20*u.AA)
    assert len(figure.axes) == 2
    figure.savefig(tmp_path/'flat_check.png')
    plt.close(figure)


def test_wavelength_gaps_and_selected_intervals_have_no_fictitious_width(tmp_path):
    path = tmp_path/'gap.fits'
    write_flat_spectrum(path, np.r_[np.arange(4000., 4005.), np.arange(4010., 4015.)])
    table, figure = flat_ab_extraction_bins({'UVB': path}, intervals={'UVB': [(4000*u.AA, 4014*u.AA)]}, max_gap={'UVB': 1.1*u.AA}, magnitude=20, bin_R=1000)
    np.testing.assert_allclose(table['covered_width'].sum(), 9*u.AA)
    assert not np.any((table['wave_low'] < 4009.5*u.AA) & (table['wave_high'] > 4004.5*u.AA))
    np.testing.assert_allclose(table['ratio'], 1)
    plt.close(figure)


def test_missing_covariance_fails_instead_of_using_diagonal_errors(tmp_path):
    path = tmp_path/'diagonal_only.fits'
    spectrum = Spectrum(m=None, spec=np.ones(10), wave=np.arange(4000., 4010.), sig=np.ones(10))
    Spectra(header=fits.Header(), data=[spectrum]).save(path)
    with pytest.raises(ValueError, match='no propagated spectral covariance'):
        flat_ab_extraction_bins({'UVB': path}, intervals={'UVB': [(4000*u.AA, 4010*u.AA)]}, max_gap={'UVB': 1.1*u.AA}, magnitude=20, bin_R=1000)


@pytest.mark.parametrize('unit,scale', [('erg s-1 cm-2 nm-1', 10.), ('erg s-1 cm-2 Angstrom-1', 1.)])
def test_flux_and_covariance_units_are_converted_together(tmp_path, unit, scale):
    path = tmp_path/'unit_check.fits'
    spectrum = write_flat_spectrum(path, np.arange(4000., 4021.))
    spectrum.spec *= scale
    spectrum.sig *= scale
    spectrum.covariance *= scale**2
    header = fits.Header({'BUNIT': unit, 'COVUNIT': (u.Unit(unit)**2).to_string('fits')})
    Spectra(header=header, data=[spectrum]).save(path)
    table, figure = flat_ab_extraction_bins({'BLUE': path}, intervals={'BLUE': [(4000*u.AA, 4020*u.AA)]}, max_gap={'BLUE': 1.1*u.AA}, magnitude=20, bin_R=1000)
    np.testing.assert_allclose(table['ratio'], 1)
    assert table['uncertainty'][0].value < 1e-18
    plt.close(figure)


def test_missing_or_incompatible_flux_units_fail(tmp_path):
    path = tmp_path/'bad_units.fits'
    spectrum = write_flat_spectrum(path, np.arange(4000., 4021.))
    arguments = dict(intervals={'BLUE': [(4000*u.AA, 4020*u.AA)]}, max_gap={'BLUE': 1.1*u.AA}, magnitude=20, bin_R=1000)
    Spectra(header=fits.Header(), data=[spectrum]).save(path)
    with pytest.raises(KeyError, match='BUNIT'):
        flat_ab_extraction_bins({'BLUE': path}, **arguments)
    Spectra(header=fits.Header({'BUNIT': 'electron', 'COVUNIT': 'electron2'}), data=[spectrum]).save(path)
    with pytest.raises(u.UnitConversionError):
        flat_ab_extraction_bins({'BLUE': path}, **arguments)


def test_band_integral_keeps_cross_gap_covariance_and_real_coverage(tmp_path):
    path = tmp_path/'band.fits'
    spectrum = write_flat_spectrum(path, np.r_[np.arange(4000., 4005.), np.arange(4010., 4015.)], factor=1.2)
    table = flat_ab_band_integrals({'BLUE': path}, intervals={'BLUE': [(4000*u.AA, 4014*u.AA)]}, max_gap={'BLUE': 1.1*u.AA}, magnitude=20)
    row, = table
    widths = np.array([.5, 1, 1, 1, 1, 1, 1, 1, 1, .5])
    weights = widths/widths.sum()
    expected_variance = weights @ (spectrum.covariance @ weights)
    without_cross_gap = expected_variance - 2*weights[4]*weights[5]*spectrum.covariance[4, 5]
    np.testing.assert_allclose(row['covered_width'], 9*u.AA)
    np.testing.assert_allclose(row['ratio'], 1.2)
    np.testing.assert_allclose(row['uncertainty'].value**2, expected_variance)
    assert row['uncertainty'].value**2 > without_cross_gap
    np.testing.assert_allclose(row['ratio_uncertainty'], row['uncertainty']/row['expected_flux'])


def test_band_intervals_are_explicit_and_cannot_overlap(tmp_path):
    path = tmp_path/'windows.fits'
    write_flat_spectrum(path, np.arange(4000., 4021.))
    arguments = dict(max_gap={'BLUE': 1.1*u.AA}, magnitude=20)
    table = flat_ab_band_integrals({'BLUE': path}, intervals={'BLUE': [(4001*u.AA, 4004*u.AA), (4010*u.AA, 4012*u.AA)]}, **arguments)
    np.testing.assert_allclose(table['covered_width'][0], 5*u.AA)
    np.testing.assert_allclose(table['ratio'], 1)
    with pytest.raises(ValueError, match='without overlap'):
        flat_ab_band_integrals({'BLUE': path}, intervals={'BLUE': [(4000*u.AA, 4010*u.AA), (4005*u.AA, 4015*u.AA)]}, **arguments)
    with pytest.raises(ValueError, match='no samples'):
        flat_ab_band_integrals({'BLUE': path}, intervals={'BLUE': [(5000*u.AA, 5010*u.AA)]}, **arguments)


def test_band_flux_and_covariance_units_are_converted(tmp_path):
    path = tmp_path/'band_units.fits'
    spectrum = write_flat_spectrum(path, np.arange(4000., 4021.))
    arguments = dict(intervals={'BLUE': [(4000*u.AA, 4020*u.AA)]}, max_gap={'BLUE': 1.1*u.AA}, magnitude=20)
    original = flat_ab_band_integrals({'BLUE': path}, **arguments)
    spectrum.spec *= 10
    spectrum.sig *= 10
    spectrum.covariance *= 100
    header = fits.Header({'BUNIT': 'erg s-1 cm-2 nm-1', 'COVUNIT': 'erg2 s-2 cm-4 nm-2'})
    Spectra(header=header, data=[spectrum]).save(path)
    converted = flat_ab_band_integrals({'BLUE': path}, **arguments)
    np.testing.assert_allclose(converted['flux'], original['flux'])
    np.testing.assert_allclose(converted['uncertainty'], original['uncertainty'])
