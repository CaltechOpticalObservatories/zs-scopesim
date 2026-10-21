"""Check saved ZShooter products and integrate selected windows with their covariance.

This checks the one-hour notebook baseline. The reference is its two-component
Kasen transient. Extracted science
also contains host light, so scene/transient ratios are not pure flux closure.
Run after the notebook; select completed epochs with ``--phases 1.5 7.5 20``.
"""

from pathlib import Path
import argparse
import gc

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from synphot.units import convert_flux
from pyreduce.spectra import Spectra
from zsdrp.steps import TraceWrapper
from zs_scopesim_tools.sources import kilonova_spectra


repo = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reduced_dir', type=Path, nargs='?', default=repo / 'notebooks/01_Kilonovae/kilonova_X-1.3_NS-0.7_dark_slit-0.7_ao-0_optimal_variance/reduced')
parser.add_argument('--phases', type=float, nargs='+', default=[1.5, 7.5])
parser.add_argument('--output', type=Path, default=repo / 'validation/xshooter/data/kilonova_reduction/zshooter_kilonova_product_check.csv')
args = parser.parse_args()

windows = {'BLUE': (3500, 4000), 'GREEN': (4800, 6000), 'RED': (8000, 8800), 'YJ': (12000, 13000), 'H': (15500, 17000), 'K': (21000, 22000)}
models = kilonova_spectra(['knova_d1_n10_m0.025_vk0.30_Xlan1e-4.0.h5', 'knova_d1_n10_m0.040_vk0.15_Xlan1e-1.5.h5'], phases=args.phases, redshift=.0098)
flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
rows = []
for phase in args.phases:
    for arm, (lo, hi) in windows.items():
        path = args.reduced_dir / f'spectra_KN_{phase}D_SUB_{arm}_fluxcal.fits.gz'
        if not path.exists():
            path = path.with_suffix('')
        product = Spectra.read(path, raw=True)
        spectrum, = product.data
        assert u.Unit(product.header['BUNIT']) == flux_unit
        assert u.Unit(product.header['COVUNIT']) == flux_unit**2
        assert product.header['EXPTIME'] == 3600
        with fits.open(path) as hdul:
            assert u.Unit(hdul['COV0000'].columns['DATA'].unit) == flux_unit**2
        wave, flux, sig, cov = spectrum.wave, spectrum.spec, spectrum.sig, spectrum.covariance
        assert np.all(np.isfinite(wave)) and np.all(np.diff(wave) > 0)
        assert np.all(np.isfinite(flux)) and np.all(np.isfinite(sig)) and np.all(sig > 0)
        assert np.all(np.isfinite(cov.data))
        np.testing.assert_allclose(sig**2, cov.diagonal(), rtol=1e-12, atol=0)
        asymmetry = cov - cov.T
        assert not asymmetry.nnz or np.max(np.abs(asymmetry.data)) < 1e-12 * np.max(cov.diagonal())

        trace_path = args.reduced_dir / f'trace_{arm}.fits.gz'
        if not trace_path.exists():
            trace_path = trace_path.with_suffix('')
        traces = [trace for trace in TraceWrapper.load(trace_path) if not trace.invalid]
        with np.load(args.reduced_dir / f'wavecal_{arm}.npz') as calibration:
            native_steps = np.concatenate([np.diff(cal_wave[trace.column_range[0]:trace.column_range[1]]) for cal_wave, trace in zip(calibration['wave_img'], traces, strict=True)])
            native_gap = np.max(np.abs(native_steps))
            orders = calibration['orders']
            np.testing.assert_array_equal(orders, [trace.m for trace in traces])
        widths = np.zeros(len(wave))
        # Native pixel boundaries are adjacent-center midpoints within each
        # supported segment; gaps contribute no width or model flux.
        for index in np.split(np.arange(len(wave)), np.flatnonzero(np.diff(wave) > native_gap) + 1):
            if len(index) < 2:
                raise ValueError(f'{arm} has an isolated sample without defined pixel width: {index}')
            centers = wave[index]
            edges = np.r_[centers[0] - (centers[1]-centers[0])/2, (centers[:-1]+centers[1:])/2, centers[-1] + (centers[-1]-centers[-2])/2]
            widths[index] = np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
        selected = np.flatnonzero(widths)
        weights = widths[selected] / widths.sum()
        measured = weights @ flux[selected]
        uncertainty = np.sqrt(weights @ (cov[selected][:, selected] @ weights))
        truth = convert_flux(wave[selected] * u.AA, models[phase](wave[selected] * u.AA), 'FLAM').value
        transient_mean = weights @ truth
        if transient_mean == 0:
            ratio = ratio_uncertainty = np.ma.masked
            print(f'{phase:g}-day {arm}: the input transient is exactly zero in this window; scene flux retains host light and noise. The two scene/transient ratios are undefined and masked.')
        else:
            ratio, ratio_uncertainty = measured/transient_mean, uncertainty/transient_mean
        band = (wave >= lo) & (wave <= hi)
        sigma_median, sigma_p99, sigma_max = np.percentile(sig[band], [50, 99, 100])
        rows.append((phase, arm, len(wave), wave[0], wave[-1], min(orders), max(orders), len(orders), lo, hi, widths.sum(), measured, uncertainty, transient_mean, ratio, ratio_uncertainty, measured/uncertainty, np.mean(flux[band] < 0), sigma_median, sigma_p99, sigma_max))
        del product, spectrum, cov, asymmetry
        gc.collect()

table = Table(rows=rows, masked=True, names=['phase_days', 'arm', 'samples', 'wave_min_A', 'wave_max_A', 'order_min', 'order_max', 'orders', 'window_low_A', 'window_high_A', 'covered_width_A', 'scene_mean_FLAM', 'sigma_mean_FLAM', 'transient_mean_FLAM', 'scene_over_transient', 'sigma_scene_over_transient', 'scene_band_snr', 'negative_sample_fraction', 'native_sigma_median_FLAM', 'native_sigma_p99_FLAM', 'native_sigma_max_FLAM'])
args.output.parent.mkdir(parents=True, exist_ok=True)
table.write(args.output, format='ascii.csv', overwrite=True)
table[['phase_days', 'arm', 'scene_over_transient', 'sigma_scene_over_transient', 'scene_band_snr']].pprint(max_width=-1)
print(f'Saved {args.output}')
