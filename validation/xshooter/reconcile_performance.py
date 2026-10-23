"""Export the saved performance tables from explicit notebook snapshots.

This reads already executed evidence; it does not execute or edit a notebook.
Use main, the preserved user rerun, and the corrected execution as separate inputs.
"""
import argparse
import json
import re
from io import StringIO
from pathlib import Path
import pandas as pd


TABLE_CELLS = {
    'a55b6247': 'slit_033',
    '38ba78232aade192': 'slit_070',
    'c7717613': 'throughput',
    '251e09e1': 'source_sky_rates',
    '7ded78cd': 'photon_budget',
    'b77fc4f5': 'detector_noise',
    'b89c0d370d46d895': 'frame_source_counts',
    '1b040c3b07df4b57': 'native_extraction',
    'ef0d24daa056999b': 'natural_ao_limits',
    '23a964d137134d93': 'sensitivity_summary',
    '83a23129825b6a9f': 'detector_alternatives',
    'detector-sensitivity-summary': 'detector_trade',
    'f541c50cd8536edc': 'thermal_budget',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='+', help='state=path to preserved, executed notebook JSON')
    args = parser.parse_args()
    output = Path(__file__).parent/'data/reconciliation'
    output.mkdir(exist_ok=True)
    for item in args.snapshots:
        state, filename = item.split('=', 1)
        notebook = json.loads(Path(filename).read_text())
        for cell in notebook['cells']:
            if cell['id'] not in TABLE_CELLS:
                continue
            tables = []
            for result in cell['outputs']:
                if 'data' not in result or 'text/html' not in result['data']:
                    continue
                html = ''.join(result['data']['text/html'])
                if '<table' in html:
                    tables.extend(pd.read_html(StringIO(html)))
            for index, table in enumerate(tables):
                table.to_csv(output/f'{state}_{TABLE_CELLS[cell["id"]]}_{index}.csv', index=False)
        noise = pd.read_csv(output/f'{state}_detector_noise_0.csv').set_index('channel')
        peak_cell = next(cell for cell in notebook['cells'] if cell['id'] == '1b040c3b07df4b57')
        peak_text = ''.join(''.join(result['text']) for result in peak_cell['outputs'] if result['output_type'] == 'stream')
        peaks = re.findall(r'(\w+) pixel \((\d+), (\d+)\): signal=([\d.]+) e, variance=([\d.]+) e2, S/N=([\d.]+)', peak_text)
        rows = []
        for channel, x, y, signal, variance, snr in peaks:
            signal, variance = float(signal), float(variance)
            detector = noise.loc[channel]
            # The printed variance is S + 2(B + D + RN**2), for an independent sky frame.
            sky = (variance-signal)/2-detector.dark_current_e_pix-detector.read_noise_e_rms**2
            rows.append((channel, int(x), int(y), signal, sky, detector.dark_current_e_pix, detector.read_noise_e_rms**2, variance, float(snr)))
        pd.DataFrame(rows, columns=['channel', 'pixel_x', 'pixel_y', 'source_e', 'sky_e', 'dark_e', 'read_variance_e2', 'subtracted_variance_e2', 'snr']).to_csv(output/f'{state}_peak_pixel_noise.csv', index=False)
        print(f'{state}: saved displayed tables from {filename}')
