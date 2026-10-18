"""Read the six-channel ZShooter archive's prescription; this does not ray trace it."""

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table


def camera_focal_length(surfaces, glasses, first, last, wave_um):
    """Paraxial air-to-air focal length from the camera surfaces and glass catalog."""
    matrix = np.eye(2)
    n_before = 1.
    for number in range(first, last + 1):
        surface = surfaces[number]
        curvature = float(surface['CURV'][0])
        if surface['TYPE'][0] == 'EVENASPH':
            curvature += 2*float(surface['PARM', 1][1])
        if 'GLAS' in surface:
            formula, coefficients = glasses[surface['GLAS'][0]]
            # This archive uses Sellmeier 1 and 3: three or four coefficient pairs.
            assert formula in (2, 6), formula
            pairs = coefficients[:6 if formula == 2 else 8].reshape(-1, 2)
            n_after = np.sqrt(1 + np.sum(pairs[:, 0]*wave_um**2/(wave_um**2 - pairs[:, 1])))
        else:
            n_after = 1.  # No GLAS record denotes air between camera elements.
        # State is (height, n*slope), with signed Zemax curvature and thickness in mm.
        refraction = np.array([[1, 0], [-(n_after - n_before)*curvature, 1]])
        translation = np.array([[1, float(surface['DISZ'][0])/n_after], [0, 1]])
        matrix = translation @ refraction @ matrix
        n_before = n_after
    # Some folded channels propagate toward negative local z.
    return abs(-1/matrix[1, 0])


def inspect_archive(archive, output):
    from zmxtools import zar

    archive, output = Path(archive), Path(output)
    files = {Path(entry.file_name.replace('\\', '/')).name: entry.unpacked_contents for entry in zar.read(archive)}
    lines = files[archive.with_suffix('.zmx').name].decode('utf-16').splitlines()
    surfaces, parameters, waves, apertures, tilts = {}, {}, {}, {}, {}
    for line in lines:
        parts = line.split()
        if line.startswith('SURF '):
            number = int(parts[1])
            surfaces[number] = {}
        elif line.startswith('  ') and parts:
            key = (parts[0], int(parts[1])) if parts[0] == 'PARM' else parts[0]
            surfaces[number][key] = parts[1:]
        elif line.startswith('PRAM '):
            parameters[int(parts[1]), int(parts[2]), int(parts[5])] = float(parts[3])
        elif line.startswith('WAVE '):
            waves[int(parts[2]), int(parts[1])] = float(parts[3])
        elif line.startswith(('APMN ', 'APMX ')):
            apertures[parts[0], int(parts[2])] = float(parts[3])
        elif line.startswith('CBTX '):
            tilts[int(parts[1]), int(parts[2])] = float(parts[3])
        elif line.startswith('FTYP '):
            n_fields, n_waves = int(parts[3]), int(parts[4])
        elif line.startswith('PWAV '):
            primary_wave = int(parts[1])

    glasses = {}
    for line in files['ZSHOOTER.AGF'].decode('utf-16').splitlines():
        parts = line.split()
        if parts[0] == 'NM':
            name, formula = parts[1], int(parts[2])
        elif parts[0] == 'CD':
            glasses[name] = formula, np.array([float(value) for value in parts[1:]])

    channels, wavelengths, sampling = [], [], []
    for index, band in enumerate(['b', 'g', 'r', 'yj', 'h', 'k']):
        first_config = 3*index + 1
        config = first_config + 1
        min_order = int(parameters[61, first_config, 2])
        max_order = int(parameters[61, first_config + 2, 2])
        wave = waves[config, primary_wave]
        focal_length = camera_focal_length(surfaces, glasses, 95 + 12*index, 104 + 12*index, wave)
        channels.append({
            'channel': band, 'min_order': min_order, 'max_order': max_order,
            'echelle_grooves_per_mm': 1000*parameters[61, config, 1],
            'echelle_surface_tilt_deg': abs(tilts[61, config]),
            'cx_grooves_per_mm': 1000*parameters[88, config, 1],
            'cx_coordinate_tilt_x_deg': parameters[85, config, 3],
            'camera_center_tilt_y_deg': parameters[93, config, 4],
            'detector_coordinate_rotation_z_deg': parameters[166, config, 5],
            'image_aperture_x_mm': 2*apertures['APMN', config],
            'image_aperture_y_mm': 2*apertures['APMX', config],
            'camera_reference_wave_nm': 1000*wave, 'camera_paraxial_f_mm': focal_length,
        })
        for config in range(first_config, first_config + 3):
            order = int(parameters[61, config, 2])
            values = np.array([waves[config, slot] for slot in range(1, n_waves + 1)])
            for slot, value in enumerate(values, 1):
                wavelengths.append((band, order, slot, 1000*value))
            step = (values[-1] - values[0])/(n_waves - 1)
            # Test the observed 11-of-12 pattern; the extrapolated endpoint is not a traced ray.
            red_edge = values[-1] + step
            center = (values[0] + red_edge)/2
            full_fsr = center/order
            sampling.append((band, order, 1000*values[0], 1000*values[-1], 1000*step, 1000*red_edge, 1000*full_fsr, (values[-1] - values[0])/full_fsr))
            np.testing.assert_allclose(np.diff(values), step, atol=1e-6, rtol=0)  # Inputs are rounded to 1 pm.

    output.mkdir(parents=True, exist_ok=True)
    Table(rows=channels).write(output / 'zemax_channels.csv', overwrite=True)
    Table(rows=wavelengths, names=['channel', 'order', 'sample', 'wavelength_nm']).write(output / 'zemax_wavelengths.csv', overwrite=True)
    Table(rows=sampling, names=['channel', 'order', 'first_wave_nm', 'last_wave_nm', 'step_nm', 'extrapolated_twelfth_wave_nm', 'inferred_full_fsr_nm', 'listed_fsr_fraction']).write(output / 'zemax_sampling.csv', overwrite=True)
    print(f'{archive.name}: {n_fields} active fields, {n_waves} active wavelengths; exported six channels and {len(wavelengths)} wavelength samples.')
    return channels, sampling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect_archive(args.archive, args.output)
