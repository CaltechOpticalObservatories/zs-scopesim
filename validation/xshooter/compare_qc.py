"""Compare independent ESO QC efficiencies with the response-calibration epoch.

Inputs are ASCII exports of xshooter_fluxeff from the public ESO QC1 database.
These are extinction-corrected telescope+instrument+detector efficiencies, not
the flat-field-dependent RESPONSE function or a test of simulated slit losses.
"""
import argparse
import io
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('reference', type=Path)
parser.add_argument('recent', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
frames = []
for path in [args.reference, args.recent]:
    lines = path.read_text().splitlines()
    names = [item.strip() for item in lines[0].split('!') if item.strip()]
    frames.append(pd.read_csv(io.StringIO('\n'.join(lines[2:])), sep=r'\s+', names=names, na_values=['None']))
reference, recent = frames
rows = []
for arm in ['UVB', 'VIS', 'NIR']:
    ref = reference.loc[reference.arm == arm].iloc[0]
    sample = recent.loc[recent.arm == arm]
    for band in range(1, 4):
        if ref[f'end{band}'] == 0:
            continue  # This arm has no third QC interval.
        assert (sample[f'start{band}'] == ref[f'start{band}']).all()
        assert (sample[f'end{band}'] == ref[f'end{band}']).all()
        efficiency = sample[f'eff{band}'].to_numpy()
        assert np.isfinite(efficiency).all()
        q10, median, q90 = np.quantile(efficiency, [.1, .5, .9])
        rows.append([arm, ref[f'start{band}'], ref[f'end{band}'], len(sample), ref[f'eff{band}'], q10, median, q90, median/ref[f'eff{band}']])
result = pd.DataFrame(rows, columns=['arm', 'wave_min_nm', 'wave_max_nm', 'observations', 'reference_efficiency', 'recent_p10', 'recent_median', 'recent_p90', 'median_over_reference'])
result.to_csv(args.output, index=False, float_format='%.6g')
print(result.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('Recent sample marked photometric:', (recent.photometric == 'Y').sum(), '/', len(recent))
