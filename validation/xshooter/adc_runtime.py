"""Time flat-AB observations with corrected or legacy ADC ordering.

Run once per ordering in fresh processes, e.g. --ordering corrected --output corrected.csv.
The legacy callback reproduces post-slit shifting without changing the installed code.
Both cases use identical fresh instruments, a centered flat AB20 source, airmass 1.3,
seeing 0.7 arcsec, and the fixed kilonova epoch. Atmospheric transmission and emission are
omitted, with other optical effects retained. Setup/network/readout time and the
host/atmospheric-background workloads are not measured here.
"""
import argparse
from pathlib import Path
from subprocess import check_output
from time import perf_counter

import astropy.units as u
import pandas as pd
import scopesim as sim
from scopesim.effects import MoffatPSF
from synphot import SourceSpectrum, ConstFlux1D
from zs_scopesim_tools.helpers import configure_irdb_path, warning_prevent_sync_alt_ra_dec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ordering', choices=['corrected', 'legacy'], required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
revision = check_output(['git', '-C', str(Path(sim.__file__).parent), 'rev-parse', 'HEAD'], text=True).strip()
configure_irdb_path()
sim.utils.set_console_log_level('ERROR')

if args.ordering == 'legacy':
    native_psf = MoffatPSF._apply_convolution

    def legacy_apply(self, obj, spatial_shifts=()):
        native_psf(self, obj)
        for effect in spatial_shifts:
            effect.apply_to(obj)

    MoffatPSF._apply_convolution = legacy_apply

rows = []
for instrument in ['ZShooter_v2', 'XSHOOTER']:
    commands = sim.UserCommands(use_instrument=instrument, set_modes=['SPEC'])
    for key, value in {'!OBS.airmass': 1.3, '!OBS.seeing': .7, '!OBS.brightness': 'dark', '!OBS.mjdobs': 60310.25}.items():
        commands[key] = value
    if instrument == 'ZShooter_v2':
        for key, value in {'!INST.vis_curr_slit': .7, '!INST.nir_curr_slit': .7, '!ATMO.ao_enabled': False, '!INST.include_diffuse_emissivity': False, '!SIM.spectral.spectral_bin_width': 1e-5}.items():
            commands[key] = value
    warning_prevent_sync_alt_ra_dec(commands)
    for document in commands.yaml_dicts:
        if 'effects' in document:
            document['effects'] = [effect for effect in document['effects'] if effect['class'] not in ['TAPASTERCurve', 'SkycalcTERCurve', 'SkyBackgroundTERCurve', 'PalaceAirglowEmission']]
    train = sim.OpticalTrain(commands)
    source = sim.Source(x=[0], y=[0], ref=[0], weight=[1], spectra=[SourceSpectrum(ConstFlux1D, amplitude=20*u.ABmag)])
    start = perf_counter()
    train.observe(source, update=False)
    elapsed = perf_counter() - start
    rows.append((instrument, args.ordering, revision, len(train._last_fovs), elapsed))
    print(instrument, args.ordering, f'{elapsed:.3f}s', flush=True)
    del train
pd.DataFrame(rows, columns=['instrument', 'ordering', 'scopesim_revision', 'fov_count', 'observation_s']).to_csv(args.output, index=False)
