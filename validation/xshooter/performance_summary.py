"""Compare like-for-like saved tables from the three performance executions."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    output = Path(__file__).parent/'data/reconciliation'
    channels = ['B', 'G', 'R', 'YJ', 'H', 'K']
    rows = []
    for state in ('main', 'rerun', 'corrected'):
        counts = pd.read_csv(output/f'{state}_frame_source_counts_0.csv').set_index('channel')
        limits = pd.read_csv(output/f'{state}_natural_ao_limits_0.csv')
        limits = limits[limits['case'] == '0.70" slit'].set_index(['ch', 'bin'])
        noise = pd.read_csv(output/f'{state}_detector_noise_0.csv').set_index('channel')
        for channel in channels:
            native = limits.loc[(channel, 1)]
            binned = limits.loc[(channel, 18)]
            rows.append((state, channel, counts.loc[channel, 'sum_delta_e'], native['R med'], native['med mAB5'], binned['R med'], binned['med mAB5'], noise.loc[channel, 'read_noise_e_rms'], noise.loc[channel, 'dark_current_e_pix']))
    result = pd.DataFrame(rows, columns=['state', 'channel', 'source_e_frame', 'native_R', 'native_m5_AB', 'bin18_R', 'bin18_m5_AB', 'read_noise_e_rms', 'dark_e_pixel'])
    result.to_csv(output/'three_state_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    baseline = result[result.state == 'main'].set_index('channel')
    for state, label, marker in [('main', 'committed main', 'o'), ('rerun', 'preserved rerun', 's'), ('corrected', 'corrected', '^')]:
        data = result[result.state == state].set_index('channel').loc[channels]
        axes[0].plot(channels, data.source_e_frame/baseline.source_e_frame, marker=marker, label=label)
        axes[1].plot(channels, data.native_m5_AB, marker=marker)
        axes[2].plot(channels, data.bin18_m5_AB, marker=marker)
    axes[0].set(ylabel='Full-frame source electrons / main', title='AB 20; 3600 s; .70 arcsec')
    axes[1].set(ylabel='Median order-peak limiting AB magnitude', title='Native extraction; OH disabled')
    axes[2].set(ylabel='Median order-peak limiting AB magnitude', title='18 native elements; OH disabled')
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlabel('Channel')
    fig.savefig(output/'three_state_summary.png', dpi=160)
    print(result.to_string(index=False))
