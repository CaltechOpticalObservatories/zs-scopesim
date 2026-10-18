"""Separate numerical refinement from genuine native-pixel phase response."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output = Path(__file__).parent/'data'
errors = pd.read_csv(output/'line_convergence_errors.csv')
samples = pd.read_csv(output/'line_convergence.csv')
fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
for index, ((instrument, channel), data) in enumerate(errors.groupby(['instrument', 'channel'], sort=False)):
    label = f'{instrument} {channel}'
    color = f'C{index}'
    for ax, column in zip(axes[0], ('flux_fraction', 'width_fraction', 'centroid_pixel')):
        ax.scatter(data.pixel_phase, data[column]*(1 if column == 'centroid_pixel' else 100), s=14, label=label, color=color)
    reference = samples[(samples.instrument == instrument) & (samples.channel == channel) & (samples.pixel_quadrature == 16)]
    for fwhm, group in reference.groupby('fwhm_arcsec'):
        group = group.sort_values('pixel_phase')
        phase_zero = group.iloc[0]
        style = '-' if fwhm == .65 else '--'
        axes[1, 0].plot(group.pixel_phase, 100*(group.flux/phase_zero.flux-1), style, lw=.8, color=color)
        axes[1, 1].plot(group.pixel_phase, 100*(group.sigma_um/phase_zero.sigma_um-1), style, lw=.8, color=color)
        bias = (group.centroid_um-phase_zero.centroid_um)/group.pixel_um-group.pixel_phase
        axes[1, 2].plot(group.pixel_phase, bias, style, lw=.8, color=color)
for ax, title in zip(axes[0], ('Flux refinement error [%]', 'RMS width refinement error [%]', 'Centroid refinement error [pixel]')):
    ax.set_title(title)
    ax.axhline(0, color='.6', lw=.7)
for ax, title in zip(axes[1], ('Flux phase response [%]', 'RMS width phase response [%]', 'Centroid phase response [pixel]')):
    ax.set_title(title)
for ax in axes.flat:
    ax.set_xlabel('Native detector pixel phase')
    ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=6, ncol=2)
fig.suptitle('Upper: default versus refined integration; lower: refined native-pixel measurements\n0.33 arcsec slit, +0.0825 arcsec source; solid PSF FWHM 0.65 arcsec, dashed 0.12 arcsec')
fig.savefig(output/'line_convergence.png', dpi=160)
