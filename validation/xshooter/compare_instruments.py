"""Compare the noiseless image planes saved by Upstream_Observations.ipynb."""
from pathlib import Path
import numpy as np
import pandas as pd


def compare_instruments(directory):
    directory = Path(directory)
    rows = []
    for baseline_file in sorted(directory.glob('upstream-scopesim_*.npz')):
        case = baseline_file.stem.removeprefix('upstream-scopesim_')
        with np.load(baseline_file) as baseline, np.load(directory / f'upstream-port_{case}.npz') as port:
            for field in baseline.files:
                before, after = baseline[field], port[field]
                assert before.shape == after.shape
                before_rate, after_rate = before.sum(dtype=np.float64), after.sum(dtype=np.float64)
                assert before_rate > 0 and after_rate > 0
                l1 = np.abs(after - before).sum(dtype=np.float64) / before_rate
                moments = []
                for image in (before, after):
                    spatial_moments = []
                    for axis in (0, 1):
                        profile = image.sum(axis=axis, dtype=np.float64)
                        pixels = np.arange(profile.size)
                        center = (pixels * profile).sum() / profile.sum()
                        sigma = np.sqrt(((pixels - center) ** 2 * profile).sum() / profile.sum())
                        spatial_moments.extend((center, sigma))
                    moments.append(spatial_moments)
                dx, dsigma_x, dy, dsigma_y = np.subtract(moments[1], moments[0])
                rows.append((case, field, before_rate, after_rate, after_rate / before_rate - 1, l1, dx, dy, dsigma_x, dsigma_y))
    return pd.DataFrame(rows, columns=['instrument', 'field', 'baseline_e_s', 'port_e_s', 'relative_flux_change', 'l1_fraction', 'dx_pixel', 'dy_pixel', 'dsigma_x_pixel', 'dsigma_y_pixel'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('output')
    args = parser.parse_args()
    comparison = compare_instruments(args.directory)
    comparison.to_csv(args.output, index=False, float_format='%.8g')
    print(comparison.to_string(index=False))
