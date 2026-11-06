"""Time trace construction and one full-band flat-AB observation in the selected checkout."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.utils import iers
from synphot import SourceSpectrum
from synphot.models import ConstFlux1D
import scopesim as sim
from validation.zshooter_xdisp.geometry import build_selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--irdb', type=Path, required=True)
    parser.add_argument('--atmosphere-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sim.rc.__config__['!SIM.file.local_packages_path'] = str(args.irdb.resolve())
    sim.rc.__search_path__.append(str(args.atmosphere_data.resolve()))
    iers.conf.auto_download = False
    timings = []
    for repeat in range(1, 4):
        start = perf_counter()
        params, effect = build_selected(args.irdb)
        timings.append(('trace_construction', repeat, perf_counter() - start))
    start = perf_counter()
    # A fixed observation time makes atmosphere and PSF inputs repeatable.
    cmds = sim.UserCommands(use_instrument='ZShooter_v2', set_modes=['SPEC'], properties={'!OBS.mjdobs': '2025-01-01T10:00:00'})
    train = sim.OpticalTrain(cmds)
    timings.append(('train_setup', 1, perf_counter() - start))
    source = sim.Source(x=[0.], y=[0.], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=20*u.ABmag)])
    start = perf_counter()
    train.observe(source)
    timings.append(('observation', 1, perf_counter() - start))
    assert len(train.image_planes) == 6
    assert all(np.isfinite(plane.data).all() and plane.data.sum() > 0 for plane in train.image_planes)
    table = Table(rows=timings, names=['stage', 'repeat', 'seconds'])
    table.write(args.output, overwrite=True)
    print(table)
