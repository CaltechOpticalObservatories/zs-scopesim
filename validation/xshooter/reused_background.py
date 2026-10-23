"""Compare matched AO and natural-seeing backgrounds on one configured K order."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import scopesim as sim
from scopesim.source.source_templates import ab_spectrum
from zs_scopesim_tools.helpers import configure_irdb_path, set_scopesim_log_level, disable_background, disable_scopesim_progress_bars, warning_prevent_sync_alt_ra_dec


if __name__ == '__main__':
    configure_irdb_path()
    set_scopesim_log_level('error')
    properties = {'!OBS.airmass': 1.1, '!OBS.seeing': .65, '!OBS.brightness': 'dark', '!OBS.mjdobs': 60310.25, '!INST.vis_curr_slit': .33, '!INST.nir_curr_slit': .33, '!INST.include_diffuse_emissivity': False}
    commands = sim.UserCommands(use_instrument='ZShooter_v2', properties=properties)
    warning_prevent_sync_alt_ra_dec(commands)
    train = sim.OpticalTrain(commands)
    orders = sorted({fov.trace_id for fov in train.fov_manager.fovs if str(fov.trace_id).startswith('k_')})
    order = orders[len(orders)//2]
    train.fov_manager._fovs_list = [fov for fov in train.fov_manager.fovs if fov.trace_id == order]
    image_id = train.fov_manager.fovs[0].meta['image_plane_id']
    fresh_fovs = deepcopy(train.fov_manager.fovs)
    source = sim.Source(x=[0.], y=[0.], ref=[0], weight=[1.], spectra=[ab_spectrum(mag=20)])
    rows, images = [], {}
    for reset in (False, True):
        for ao in (False, True):
            train['seeing_psf'].meta['enable_ao'] = ao
            for lines in (False, True):
                for target in (None, source):
                    if reset:
                        train.fov_manager._fovs_list = deepcopy(fresh_fovs)
                    for plane in train.image_planes:
                        plane.data.fill(0)
                    with disable_background(None if lines else ['sky_lines'], train), disable_scopesim_progress_bars():
                        train.observe(target, update=False)
                    plane, = [plane for plane in train.image_planes if plane.id == image_id]
                    images[reset, ao, lines, target is not None] = plane.data.copy()
                background = images[reset, ao, lines, False]
                signal = images[reset, ao, lines, True] - background
                assert signal.sum() > 0
                rows.append((reset, ao, lines, background.sum(), signal.sum()))
                print(rows[-1], flush=True)
    result = pd.DataFrame(rows, columns=['fresh_fov', 'ao', 'sky_lines', 'background_e_s', 'source_e_s'])
    reused = result[~result.fresh_fov].set_index(['ao', 'sky_lines'])
    fresh = result[result.fresh_fov].set_index(['ao', 'sky_lines'])
    np.testing.assert_allclose(reused[['background_e_s', 'source_e_s']], fresh[['background_e_s', 'source_e_s']], rtol=1e-10)
    for ao in (False, True):
        np.testing.assert_allclose(reused.loc[(ao, False), 'source_e_s'], reused.loc[(ao, True), 'source_e_s'], rtol=1e-10)
    print(order, result.to_string(index=False), sep='\n')
    for reset in (False, True):
        delta = images[reset, True, False, False] - images[reset, False, False, False]
        source_only = images[reset, True, False, True] - images[reset, True, False, False]
        print('fresh', reset, 'AO-natural sky', delta.sum(), 'relative to source', delta.sum()/source_only.sum())
    result.to_csv(Path(__file__).parent / 'data/reconciliation/reused_background.csv', index=False)
