"""Detector coverage from physical intersections, independent of trace sampling."""

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from validation.zshooter_xdisp.geometry import band_coverage, minimum_vertical_slit_gap, trace_intervals


def linear_trace(samples=2, x_offset=0, tilted=False):
    wave = np.linspace(100, 110, samples)
    x = np.linspace(-2, 2, samples) + x_offset
    center_y = np.linspace(-1.5, 1.5, samples) if tilted else np.zeros(samples)
    y = center_y[None, :] + np.array([-.5, 0, .5])[:, None]
    table = Table({'wavelength': np.tile(wave, 3) * u.nm, 'x': np.tile(x, 3) * u.mm, 'y': y.ravel() * u.mm})
    hdu = fits.BinTableHDU(table)
    hdu.header['EXTNAME'] = 'b_1'
    return hdu


RECTANGLE = {'prefix': 'b', 'n_disp': 2, 'n_xdisp': 2, 'pixel_size': 1, 'min_wave': 100, 'max_wave': 110}


@pytest.mark.parametrize('samples', [2, 101])
def test_center_and_full_slit_intersections(samples):
    hdu = linear_trace(samples, tilted=True)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(), hdu])
    center, _ = band_coverage(hdul, RECTANGLE, full_slit=False)
    full, _ = band_coverage(hdul, RECTANGLE)
    assert center == pytest.approx(.5)
    assert full == pytest.approx(1/3)


def test_overlapping_orders_count_wavelength_once():
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(), linear_trace(), linear_trace(x_offset=-1)])
    fraction, gaps = band_coverage(hdul, RECTANGLE)
    assert fraction == pytest.approx(.75)
    assert gaps == [(100, 102.5)]


def test_parallel_slit_edge_outside_detector():
    hdu = linear_trace()
    hdu.data['y'] += 1
    assert trace_intervals(hdu, RECTANGLE, full_slit=True) == []
    assert trace_intervals(hdu, RECTANGLE, full_slit=False) == [(102.5, 107.5)]


def test_missing_pixel_size_fails():
    rectangle = RECTANGLE.copy()
    del rectangle['pixel_size']
    with pytest.raises(KeyError, match='pixel_size'):
        trace_intervals(linear_trace(), rectangle, full_slit=True)


def test_signed_slit_edge_separation():
    rectangle = dict(RECTANGLE, n_xdisp=8)
    first, second = linear_trace(), linear_trace()
    assert minimum_vertical_slit_gap([first, second], rectangle) == pytest.approx(-1)
    second.data['y'] += 2
    assert minimum_vertical_slit_gap([first, second], rectangle) == pytest.approx(1)
