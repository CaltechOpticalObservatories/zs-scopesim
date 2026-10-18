"""Bounded observations using real upstream IRDB configurations.

Select an engine before importing this module. The narrow wavelength interval
bounds memory; optical effects, PSF sampling and aperture geometry stay enabled.
The saved arrays are electron rates before stochastic detector effects.
"""
from pathlib import Path
import logging
import numpy as np
import scopesim as sim
from synphot import SourceSpectrum
from synphot.models import ConstFlux1D
from synphot.units import PHOTLAM


def observe_case(irdb, output, instrument, modes, lower, upper, properties, check_readout=False):
    sim.rc.__config__['!SIM.file.local_packages_path'] = str(irdb)
    properties = dict(properties)
    properties.update({'!SIM.spectral.wave_min': lower, '!SIM.spectral.wave_max': upper, '!SIM.computing.preload_field_of_views': False, '!OBS.dit': 10., '!OBS.ndit': 1})
    cmds = sim.UserCommands(use_instrument=instrument, set_modes=modes, properties=properties)
    train = sim.OpticalTrain(cmds)
    train.observe()
    sky = [plane.data.copy() for plane in train.image_planes]
    source = sim.Source(x=[0.], y=[0.], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=.01*PHOTLAM)])
    train.observe(source)
    signal = [plane.data - background for plane, background in zip(train.image_planes, sky)]
    if check_readout:
        errors = []
        handler = logging.Handler(level=logging.ERROR)
        handler.emit = lambda record: errors.append(record.getMessage())
        logger = logging.getLogger('astar.scopesim')
        logger.addHandler(handler)
        try:
            readouts = train.readout()
        finally:
            logger.removeHandler(handler)
        assert not errors, errors
        for readout in readouts:
            assert readout[0].header['INSTRUME']
            for hdu in readout[1:]:
                data = hdu.data
                if data is None:
                    continue
                if data.dtype.names is None:
                    assert np.isfinite(data).all()
                else:
                    for column in data.dtype.names:
                        assert np.isfinite(data[column]).all()
                assert hdu.header['DIT'] > 0
                assert hdu.header['NDIT'] >= 1
                assert hdu.header['EXPTIME'] == hdu.header['DIT'] * hdu.header['NDIT']
    assert sum(point.sum() for point in signal) > 0, 'The configured detector region must intercept the test source.'
    arrays = {}
    rows = []
    for i, (background, point) in enumerate(zip(sky, signal)):
        assert np.isfinite(background).all() and np.isfinite(point).all()
        arrays[f'sky_{i}'] = background
        arrays[f'point_{i}'] = point
        rows.append((i, point.shape, background.sum(), point.sum()))
    np.savez_compressed(Path(output), **arrays)
    return rows
