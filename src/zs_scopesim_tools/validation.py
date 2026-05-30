"""Validation helpers for ZShooter ScopeSim notebooks."""

from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM


def effect_name(effect: Any) -> str:
    """Return the notebook-facing name of a ScopeSim effect."""
    display_name = getattr(effect, "display_name", None)
    if display_name is not None:
        return display_name
    meta = getattr(effect, "meta", {})
    return meta.get("name", repr(effect))


def active_effects(ztrain: Any) -> list[Any]:
    """Return included effects from an optical train."""
    return [
        eff for eff in ztrain.optics_manager.all_effects
        if getattr(eff, "include", True)
    ]


def get_effect(ztrain: Any, display_name: str, *, active_only: bool = True) -> Any:
    """Fetch one active optical-train effect by display name."""
    effects = active_effects(ztrain) if active_only else ztrain.optics_manager.all_effects
    matches = [eff for eff in effects if effect_name(eff) == display_name]
    if len(matches) != 1:
        available = sorted(effect_name(eff) for eff in effects)
        raise ValueError(
            f"Expected one active effect named {display_name!r}; found "
            f"{len(matches)}. Available: {available}"
        )
    return matches[0]


def resolve_effect(effect: Any, selector_value: Any = None) -> Any:
    """Return a plain effect, or one selected entry from a SelectorWheel."""
    if not hasattr(effect, "wheel_effects"):
        return effect
    if selector_value is None:
        keys = sorted(effect.wheel_effects)
        raise ValueError(
            f"{effect_name(effect)!r} is selector-based; choose one of {keys}"
        )
    try:
        return effect.wheel_effects[selector_value]
    except KeyError as exc:
        keys = sorted(effect.wheel_effects)
        raise KeyError(
            f"{effect_name(effect)!r} has no selector value "
            f"{selector_value!r}; available {keys}"
        ) from exc


def get_arm_effect(ztrain: Any, display_name: str, aperture_id: int) -> Any:
    """Fetch one selected arm/channel effect from an optical train."""
    return resolve_effect(get_effect(ztrain, display_name), aperture_id)


def _as_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def evaluate_curve(curve: Any, wave: u.Quantity) -> np.ndarray:
    """Evaluate a synphot/ScopeSim curve on a wavelength quantity."""
    with u.set_enabled_equivalencies(u.spectral()):
        return _as_float_array(curve(wave))


def evaluate_throughput(effect: Any, wave: u.Quantity) -> np.ndarray:
    """Evaluate an effect or SpectralSurface-like object as throughput."""
    if hasattr(effect, "throughput"):
        return evaluate_curve(effect.throughput, wave)
    if hasattr(effect, "surface") and hasattr(effect.surface, "throughput"):
        return evaluate_curve(effect.surface.throughput, wave)
    raise TypeError(f"Cannot evaluate throughput for {effect!r}")


def fetch_effect_spectrum_or_transmission(
    ztrain: Any,
    display_name: str,
    attribute: str | None = "throughput",
    wave: u.Quantity | None = None,
    aperture_id: int | None = None,
) -> tuple[u.Quantity, Any]:
    """Fetch a curve-like attribute from an active effect.

    ``attribute`` may be dotted, e.g. ``"line_TER.emission"``.
    """
    effect = get_effect(ztrain, display_name)
    target = resolve_effect(effect, aperture_id) if aperture_id is not None else effect
    if attribute is not None:
        for attr in attribute.split("."):
            target = getattr(target, attr)

    if wave is not None:
        return wave.to(u.um), evaluate_curve(target, wave)
    if hasattr(target, "_get_arrays"):
        wave_out, values = target._get_arrays(wavelengths=None)
        return wave_out.to(u.um), values
    if hasattr(target, "waveset"):
        wave_out = target.waveset.to(u.um)
        return wave_out, evaluate_curve(target, wave_out)
    raise TypeError(f"Cannot fetch spectrum/transmission from {target!r}")


def plot_source(source: Any, wave: u.Quantity | None = None):
    """Plot each source field's spatial profile and spectrum."""
    import matplotlib.pyplot as plt

    wave = wave if wave is not None else np.linspace(0.3, 2.5, 1001) * u.um
    num_fields = len(source.fields)
    fig, axs = plt.subplots(
        figsize=(6, 2 * num_fields),
        nrows=num_fields,
        ncols=2,
        width_ratios=[1, 2],
        constrained_layout=True,
    )
    if num_fields == 1:
        axs = np.array([axs])
    for idx, field in enumerate(source.fields):
        ax_image, ax_spectrum = axs[idx]
        ax_image.imshow(field.data, origin="lower", cmap="viridis")
        ax_spectrum.plot(wave, field.spectrum(wave))
        if idx == 0:
            ax_image.set_title("Spatial Profile")
            ax_spectrum.set_title("Spectrum")
    return fig, axs


def default_surface_groups() -> OrderedDict[str, tuple[str, ...]]:
    """Fallback group rules for older optics lists without group metadata."""
    return OrderedDict([
        ("preoptics", ("Window", "ADC", "Derotator", "PreOpt", "Fold")),
        ("collimator", ("Col", "Mangin")),
        ("camera", ("Camera",)),
    ])


def _real_colname(name: str, colnames: list[str] | tuple[str, ...]) -> str | None:
    name_lower = name.lower()
    for colname in colnames:
        if colname.lower() == name_lower:
            return colname
    return None


def _row_scalar(row: Any, name: str) -> Any:
    value = row[name]
    return value.item() if hasattr(value, "item") else value


def _clean_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "None", "nan"}:
        return None
    return text


def surface_group_for_name(
    surface_name: str,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    """Return fallback throughput group from a surface name."""
    groups = groups or default_surface_groups()
    return next(
        (name for name, patterns in groups.items()
         if any(pattern in surface_name for pattern in patterns)),
        "other",
    )


def surface_group_for_row(
    row: Any,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    """Return explicit row throughput group, falling back to name patterns."""
    group_col = _real_colname("throughput_group", row.colnames)
    if group_col is not None:
        group_name = _clean_metadata_value(_row_scalar(row, group_col))
        if group_name is not None:
            return group_name

    name_col = _real_colname("name", row.colnames)
    surface_name = str(_row_scalar(row, name_col))
    return surface_group_for_name(surface_name, groups)


def emission_phase_for_row(row: Any, group_name: str | None = None) -> str:
    """Return explicit row emission phase, falling back conservatively."""
    phase_col = _real_colname("emission_phase", row.colnames)
    if phase_col is not None:
        phase_name = _clean_metadata_value(_row_scalar(row, phase_col))
        if phase_name is not None:
            return phase_name

    if group_name == "camera":
        return "post_disperser"
    return "pre_disperser"


def ter_property_sources(surface: Any) -> dict[str, str]:
    """Report whether TER properties are explicit or ScopeSim-inferred."""
    colnames = set(getattr(surface.table, "colnames", []))
    explicit = {
        name: name in surface.meta or name in colnames
        for name in ("transmission", "reflection", "emissivity")
    }
    sources = {}
    for name, is_explicit in explicit.items():
        if is_explicit:
            sources[name] = "explicit"
        else:
            other_names = [
                other for other, exists in explicit.items()
                if other != name and exists
            ]
            if other_names:
                sources[name] = "inferred from " + "+".join(other_names)
            else:
                sources[name] = "missing"
    return sources


def evaluate_ter_property(
    effect_or_surface: Any,
    property_name: str,
    wave: u.Quantity,
) -> np.ndarray:
    """Evaluate one TER property from a TERCurve or SpectralSurface."""
    surface = getattr(effect_or_surface, "surface", effect_or_surface)
    curve = getattr(surface, property_name)
    if curve is None:
        return np.full(wave.size, np.nan)
    return evaluate_curve(curve, wave)


def surface_list_group_throughputs(
    surface_list: Any,
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[OrderedDict[str, np.ndarray], dict[str, int]]:
    """Return grouped throughputs for one SurfaceList."""
    name_col = _real_colname("name", surface_list.table.colnames)
    action_col = _real_colname("action", surface_list.table.colnames)

    grouped: OrderedDict[str, np.ndarray] = OrderedDict()
    counts: dict[str, int] = {}

    for row in surface_list.table:
        surface_name = str(_row_scalar(row, name_col))
        action_name = str(_row_scalar(row, action_col))
        group_name = surface_group_for_row(row, groups)
        surface = surface_list.surfaces[surface_name]

        if group_name not in grouped:
            grouped[group_name] = np.ones(wave.size, dtype=float)
            counts[group_name] = 0
        grouped[group_name] *= evaluate_curve(getattr(surface, action_name), wave)
        counts[group_name] += 1

    return grouped, counts


def surface_list_emissivity_terms(
    surface_list: Any,
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    qe_values: np.ndarray | None = None,
) -> tuple[dict[str, OrderedDict[str, np.ndarray]], dict[str, int], list[dict[str, Any]]]:
    """Return grouped emissivity terms split by optical phase.

    The per-surface base contribution follows ``SurfaceList.combine_emissions``:
    a surface emissivity term is weighted by the downstream action throughputs
    that ScopeSim applies after that surface's own emission is added.

    For ``post_disperser`` terms, ``after_qe`` also applies the detector QE
    because this diffuse light is injected at the image plane instead of being
    trace-mapped as a spectral source.
    """
    name_col = _real_colname("name", surface_list.table.colnames)
    action_col = _real_colname("action", surface_list.table.colnames)

    rows: list[dict[str, Any]] = []
    for row in surface_list.table:
        surface_name = str(_row_scalar(row, name_col))
        action_name = str(_row_scalar(row, action_col))
        group_name = surface_group_for_row(row, groups)
        phase_name = emission_phase_for_row(row, group_name)
        surface = surface_list.surfaces[surface_name]
        rows.append({
            "surface_name": surface_name,
            "group": group_name,
            "emission_phase": phase_name,
            "action": action_name,
            "surface": surface,
            "action_values": evaluate_curve(getattr(surface, action_name), wave),
            "emissivity": evaluate_ter_property(surface, "emissivity", wave),
            "temperature": surface.meta.get("temperature"),
            "ter_sources": ter_property_sources(surface),
        })

    downstream = [np.ones(wave.size, dtype=float) for _ in range(len(rows) + 1)]
    for idx in range(len(rows) - 1, -1, -1):
        downstream[idx] = downstream[idx + 1] * rows[idx]["action_values"]

    phase_terms: dict[str, OrderedDict[str, np.ndarray]] = {
        "pre_disperser": OrderedDict(),
        "post_disperser": OrderedDict(),
    }
    counts: dict[str, int] = defaultdict(int)
    details: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if row["emission_phase"] == "none":
            continue

        contribution = row["emissivity"] * downstream[idx + 1]
        contribution = np.nan_to_num(
            contribution, nan=0.0, posinf=0.0, neginf=0.0,
        )
        after_qe = contribution
        qe_applied = False
        if row["emission_phase"] == "post_disperser" and qe_values is not None:
            after_qe = contribution * qe_values
            qe_applied = True

        phase_name = row["emission_phase"]
        if phase_name not in phase_terms:
            raise ValueError(
                f"Unknown emission_phase {phase_name!r} for "
                f"{row['surface_name']!r}; expected one of "
                f"{sorted(phase_terms)} or 'none'."
            )
        group_name = row["group"]
        if group_name not in phase_terms[phase_name]:
            phase_terms[phase_name][group_name] = np.zeros(wave.size, dtype=float)
        phase_terms[phase_name][group_name] += after_qe
        counts[f"{phase_name}:{group_name}"] += 1

        details.append({
            "surface": row["surface_name"],
            "group": group_name,
            "emission_phase": phase_name,
            "action": row["action"],
            "temperature": row["temperature"],
            "emissivity_source": row["ter_sources"]["emissivity"],
            "transmission_source": row["ter_sources"]["transmission"],
            "reflection_source": row["ter_sources"]["reflection"],
            "qe_applied": qe_applied,
            "peak_output_emissivity": (
                float(np.nanmax(contribution)) if contribution.size else np.nan
            ),
            "peak_after_qe": float(np.nanmax(after_qe)) if after_qe.size else np.nan,
        })

    return phase_terms, dict(counts), details


def representative_positional_qe(positional_qe: Any, wave: u.Quantity) -> float | np.ndarray:
    """Return an average positional QE factor for diffuse, non-trace light."""
    if positional_qe is None:
        return 1.0
    values = positional_qe(wave) if callable(positional_qe) else positional_qe
    values = _as_float_array(values)
    if values.shape == wave.shape:
        return values
    return float(np.nanmean(values))


def effective_diffuse_qe(
    detector_qe: Any,
    wave: u.Quantity,
    positional_qe: Callable[[u.Quantity], Any] | np.ndarray | None = None,
) -> np.ndarray:
    """Return detector QE for diffuse image-plane backgrounds.

    For ordinary detectors this is the spectral QE. For future tapered coatings,
    pass a positional QE map or callable; this function applies its average
    positional response instead of skipping QE for non-dispersed light.
    """
    spectral_qe = evaluate_throughput(detector_qe, wave)
    return spectral_qe * representative_positional_qe(positional_qe, wave)


def traces_by_aperture(trace_list: Any) -> dict[int, list[Any]]:
    """Group spectral traces by aperture id."""
    grouped: dict[int, list[Any]] = defaultdict(list)
    for trace_id, trace in trace_list.spectral_traces.items():
        grouped[int(trace.meta["aperture_id"])].append(trace)
    return {
        key: sorted(value, key=lambda trace: trace.trace_id)
        for key, value in grouped.items()
    }


def channel_label(aperture_id: int, traces: list[Any]) -> str:
    """Return display label for one aperture/channel."""
    if traces:
        prefix = traces[0].trace_id.partition("_")[0]
        return prefix.upper()
    return f"aperture {aperture_id}"


def trace_catalog_table(trace_list: Any) -> Table:
    """Return the loaded trace catalog from an in-memory SpectralTraceList."""
    rows = []
    for trace_id, trace in trace_list.spectral_traces.items():
        rows.append({
            "trace_id": trace.trace_id,
            "aperture_id": int(trace.meta["aperture_id"]),
            "image_plane_id": int(trace.meta["image_plane_id"]),
            "extension_id": int(trace.meta.get("extension_id", -1)),
            "wave_min_um": float(trace.wave_min),
            "wave_max_um": float(trace.wave_max),
        })
    return Table(rows=sorted(rows, key=lambda row: row["trace_id"]))


def fov_image_plane_counts(ztrain: Any) -> Table:
    """Return image-plane counts for the current in-memory FOV manager."""
    image_plane_ids = [
        int(fov.meta["image_plane_id"])
        for fov in ztrain.fov_manager.fovs
    ]
    unique_ids, counts = np.unique(image_plane_ids, return_counts=True)
    return Table({
        "image_plane_id": unique_ids.astype(int),
        "n_fovs": counts.astype(int),
    })


def build_emissivity_sanity_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    positional_qe_by_aperture: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Build split pre/post-disperser emissivity sanity-check data."""
    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)
    positional_qe_by_aperture = positional_qe_by_aperture or {}

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    channel_selector = get_effect(ztrain, "channel_optics_selector")
    qe_selector = get_effect(ztrain, "detector_qe_selector")
    trace_list = get_effect(ztrain, "trace_list_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = [
        int(value)
        for value in dichroic_tree.table[dichroic_tree.table.colnames[0]]
    ]
    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    details: list[dict[str, Any]] = []
    for aperture_id in aperture_ids:
        channel_optics = resolve_effect(channel_selector, aperture_id)
        detector_qe = resolve_effect(qe_selector, aperture_id)
        qe_values = effective_diffuse_qe(
            detector_qe,
            wave,
            positional_qe=positional_qe_by_aperture.get(aperture_id),
        )

        phase_terms, counts, surface_details = surface_list_emissivity_terms(
            channel_optics, wave, groups=groups, qe_values=qe_values,
        )
        pre_total = _sum_terms(phase_terms["pre_disperser"], wave.size)
        post_total = _sum_terms(phase_terms["post_disperser"], wave.size)

        label = channel_label(aperture_id, traces_for_aperture.get(aperture_id, []))
        for row in surface_details:
            row["aperture_id"] = aperture_id
            row["channel"] = label
            details.append(row)

        details.append({
            "aperture_id": aperture_id,
            "channel": label,
            "surface": effect_name(detector_qe),
            "group": "detector_qe",
            "emission_phase": "throughput_only",
            "action": detector_qe.surface.meta.get("action", "transmission"),
            "temperature": detector_qe.surface.meta.get("temperature"),
            "emissivity_source": "not used",
            "transmission_source": ter_property_sources(detector_qe.surface)["transmission"],
            "reflection_source": "not used",
            "qe_applied": True,
            "peak_output_emissivity": np.nan,
            "peak_after_qe": float(np.nanmax(qe_values)),
        })

        channels[aperture_id] = {
            "label": label,
            "pre_disperser_terms": phase_terms["pre_disperser"],
            "post_disperser_terms": phase_terms["post_disperser"],
            "emissivity_group_counts": counts,
            "pre_disperser_output_equiv": pre_total,
            "post_disperser_after_qe": post_total,
            "detector_qe": qe_values,
            "detector_qe_note": "throughput only; detector emissivity is not modeled",
        }

    return {"wave_nm": wave_nm, "channels": channels, "details": Table(rows=details)}


def evaluate_emission_density(surface: Any, wave: u.Quantity) -> u.Quantity | None:
    """Evaluate a ScopeSim surface emission curve on ``wave``."""
    emission = getattr(surface, "emission", None)
    if emission is None:
        return None
    values = emission(wave)
    if not isinstance(values, u.Quantity):
        values = values * PHOTLAM
    return values


def surface_list_post_disperser_diffuse_terms(
    surface_list: Any,
    wave: u.Quantity,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    qe_values: np.ndarray | None = None,
    emission_phase: str = "post_disperser",
) -> tuple[OrderedDict[str, u.Quantity], list[dict[str, Any]]]:
    """Return post-disperser diffuse spectra grouped by optics metadata.

    This is the physical counterpart to ``surface_list_emissivity_terms``. It
    follows ScopeSim's downstream-emission bookkeeping, but only surfaces tagged
    with ``emission_phase == "post_disperser"`` are returned as image-plane
    diffuse background candidates.
    """
    name_col = _real_colname("name", surface_list.table.colnames)
    action_col = _real_colname("action", surface_list.table.colnames)
    if name_col is None or action_col is None:
        raise ValueError("SurfaceList table must contain name and action columns.")

    rows: list[dict[str, Any]] = []
    for row in surface_list.table:
        surface_name = str(_row_scalar(row, name_col))
        action_name = str(_row_scalar(row, action_col))
        group_name = surface_group_for_row(row, groups)
        phase_name = emission_phase_for_row(row, group_name)
        if phase_name not in {"pre_disperser", "post_disperser", "none"}:
            raise ValueError(
                f"Unknown emission_phase {phase_name!r} for {surface_name!r}; "
                "expected pre_disperser, post_disperser, or none."
            )
        surface = surface_list.surfaces[surface_name]
        rows.append({
            "surface_name": surface_name,
            "group": group_name,
            "phase": phase_name,
            "action": action_name,
            "action_values": evaluate_curve(getattr(surface, action_name), wave),
            "emission_values": evaluate_emission_density(surface, wave),
            "temperature": surface.meta.get("temperature"),
        })

    downstream = [np.ones(wave.size, dtype=float) for _ in range(len(rows) + 1)]
    for idx in range(len(rows) - 1, -1, -1):
        downstream[idx] = downstream[idx + 1] * rows[idx]["action_values"]

    if qe_values is None:
        qe_values = np.ones(wave.size, dtype=float)

    grouped: OrderedDict[str, u.Quantity] = OrderedDict()
    details: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        is_diffuse = row["phase"] == emission_phase
        contribution = None
        if is_diffuse and row["emission_values"] is not None:
            contribution = row["emission_values"] * downstream[idx + 1] * qe_values
            group_name = row["group"]
            grouped[group_name] = (
                contribution if group_name not in grouped
                else grouped[group_name] + contribution
            )

        peak = np.nan
        if contribution is not None:
            peak = float(np.nanmax(_as_float_array(contribution)))
        details.append({
            "surface": row["surface_name"],
            "group": row["group"],
            "emission_phase": row["phase"],
            "action": row["action"],
            "temperature": row["temperature"],
            "included_as_diffuse": is_diffuse,
            "qe_applied": is_diffuse,
            "peak_after_qe": peak,
        })

    return grouped, details


def _image_plane_pixel_area(ztrain: Any, image_plane_id: int) -> u.Quantity:
    from scopesim.effects.illumination import image_plane_pixel_area

    return image_plane_pixel_area(
        ztrain.image_planes[image_plane_id].header,
        ztrain.cmds,
    )


def _telescope_area(ztrain: Any) -> u.Quantity:
    from scopesim.utils import from_currsys, quantify

    return quantify(from_currsys("!TEL.area", ztrain.cmds), u.m**2)


def build_post_disperser_diffuse_background_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    positional_qe_by_aperture: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Build image-plane post-disperser diffuse background data."""
    from scopesim.effects.illumination import integrate_spectral_background

    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)
    positional_qe_by_aperture = positional_qe_by_aperture or {}
    telescope_area = _telescope_area(ztrain)

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    channel_selector = get_effect(ztrain, "channel_optics_selector")
    qe_selector = get_effect(ztrain, "detector_qe_selector")
    trace_list = get_effect(ztrain, "trace_list_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = [
        int(value)
        for value in dichroic_tree.table[dichroic_tree.table.colnames[0]]
    ]
    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    details: list[dict[str, Any]] = []
    for aperture_id in aperture_ids:
        channel_optics = resolve_effect(channel_selector, aperture_id)
        detector_qe = resolve_effect(qe_selector, aperture_id)
        traces = traces_for_aperture.get(aperture_id, [])
        image_plane_id = (
            int(traces[0].meta["image_plane_id"]) if traces else aperture_id
        )
        image_pixel_area = _image_plane_pixel_area(ztrain, image_plane_id)
        qe_values = effective_diffuse_qe(
            detector_qe,
            wave,
            positional_qe=positional_qe_by_aperture.get(aperture_id),
        )
        spectra, surface_details = surface_list_post_disperser_diffuse_terms(
            channel_optics,
            wave,
            groups=groups,
            qe_values=qe_values,
        )
        rates = OrderedDict(
            (name, integrate_spectral_background(
                spectrum,
                wave,
                telescope_area=telescope_area,
                image_pixel_area=image_pixel_area,
            ))
            for name, spectrum in spectra.items()
        )
        total_spectrum = _sum_quantity_terms(spectra)
        total_rate = float(np.sum(list(rates.values()))) if rates else 0.0
        label = channel_label(aperture_id, traces)

        for row in surface_details:
            row["aperture_id"] = aperture_id
            row["image_plane_id"] = image_plane_id
            row["channel"] = label
            details.append(row)

        channels[aperture_id] = {
            "label": label,
            "image_plane_id": image_plane_id,
            "spectra": spectra,
            "total_spectrum": total_spectrum,
            "rates_ph_s_pix": rates,
            "total_rate_ph_s_pix": total_rate,
            "detector_qe": qe_values,
            "pixel_area": image_pixel_area,
            "telescope_area": telescope_area,
            "detector_qe_note": "throughput only; detector emissivity is not modeled",
        }

    return {"wave_nm": wave_nm, "channels": channels, "details": Table(rows=details)}


def post_disperser_diffuse_effect_consistency_table(
    ztrain: Any,
    helper_data: Mapping[str, Any],
    *,
    effect_display_name: str = "post_echelle_diffuse_background_selector",
    match_effect_grid: bool = True,
) -> Table:
    """Compare notebook helper rates to the configured image-plane effect.

    ``helper_data`` is normally the output of
    :func:`build_post_disperser_diffuse_background_data`. The ``matched`` rate
    is recomputed on the effect's own wavelength grid when possible, so the
    table separates wavelength-sampling differences from real wiring problems.
    """
    selector = get_effect(ztrain, effect_display_name, active_only=False)
    matched_data_by_image_plane = {}
    if match_effect_grid:
        matched_data_by_image_plane = _matched_post_diffuse_data_by_image_plane(
            ztrain, helper_data, selector,
        )

    rows = []
    for aperture_id, channel in helper_data["channels"].items():
        image_plane_id = int(channel["image_plane_id"])
        effect = resolve_effect(selector, image_plane_id)
        effect_rate = float(effect.background_value(ztrain.image_planes[image_plane_id]))
        helper_rate = float(channel["total_rate_ph_s_pix"])
        matched_channel = matched_data_by_image_plane.get(image_plane_id, {}).get(
            "channels", {},
        ).get(aperture_id)
        matched_rate = (
            float(matched_channel["total_rate_ph_s_pix"])
            if matched_channel is not None
            else np.nan
        )
        rows.append({
            "aperture_id": int(aperture_id),
            "channel": channel["label"],
            "image_plane_id": image_plane_id,
            "effect_class": effect.__class__.__name__,
            "effect_included": bool(getattr(effect, "include", True)),
            "surface_file": effect.meta.get("filename", ""),
            "detector_qe_file": effect.meta.get("detector_qe_filename", ""),
            "pixel_area_arcsec2": channel["pixel_area"].to_value(u.arcsec**2),
            "helper_rate_ph_s_pix": helper_rate,
            "effect_rate_ph_s_pix": effect_rate,
            "helper_rel_delta": _relative_delta(helper_rate, effect_rate),
            "matched_helper_rate_ph_s_pix": matched_rate,
            "matched_rel_delta": _relative_delta(matched_rate, effect_rate),
        })

    return Table(rows=rows)


def validate_post_disperser_diffuse_effect_consistency(
    table: Table,
    *,
    rtol: float = 1e-6,
    use_matched: bool = True,
) -> None:
    """Validate helper/effect agreement for post-disperser diffuse background."""
    delta_col = "matched_rel_delta" if use_matched else "helper_rel_delta"
    finite = np.isfinite(table[delta_col])
    if not np.all(finite):
        bad = table[~finite]
        raise ValueError(f"Non-finite {delta_col} rows: {bad}")

    abs_delta = np.abs(np.asarray(table[delta_col], dtype=float))
    if np.nanmax(abs_delta) > rtol:
        bad = table[abs_delta > rtol]
        raise ValueError(
            f"Post-disperser diffuse helper/effect mismatch above {rtol}: {bad}"
        )


def _matched_post_diffuse_data_by_image_plane(
    ztrain: Any,
    helper_data: Mapping[str, Any],
    selector: Any,
) -> dict[int, Mapping[str, Any]]:
    matched = {}
    cache = {}
    for channel in helper_data["channels"].values():
        image_plane_id = int(channel["image_plane_id"])
        effect = resolve_effect(selector, image_plane_id)
        if not hasattr(effect, "_waveset"):
            continue
        wave_nm = effect._waveset().to(u.nm)
        cache_key = tuple(np.round(wave_nm.to_value(u.nm), 12))
        if cache_key not in cache:
            cache[cache_key] = build_post_disperser_diffuse_background_data(
                ztrain, wave_nm=wave_nm,
            )
        matched[image_plane_id] = cache[cache_key]
    return matched


def _relative_delta(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    if reference == 0:
        return 0.0 if value == 0 else np.inf
    return (value - reference) / reference


def _sum_quantity_terms(terms: Mapping[str, u.Quantity]) -> u.Quantity | None:
    total = None
    for values in terms.values():
        total = values if total is None else total + values
    return total


def validate_post_disperser_diffuse_background_data(data: Mapping[str, Any]) -> None:
    """Validate post-disperser diffuse background helper output."""
    for aperture_id, channel in data["channels"].items():
        if not channel["spectra"]:
            raise ValueError(
                f"No post-disperser diffuse spectra for aperture_id={aperture_id}"
            )
        if channel["total_rate_ph_s_pix"] < -1e-12:
            raise ValueError(
                f"Negative diffuse background for aperture_id={aperture_id}: "
                f"{channel['total_rate_ph_s_pix']}"
            )
        if "detector_emissivity" in channel:
            raise ValueError("Detector emissivity must not be included.")


def _plot_quantity_values(values: u.Quantity) -> np.ndarray:
    try:
        return values.to_value(PHOTLAM)
    except Exception:
        return _as_float_array(values)


def plot_post_disperser_diffuse_background(data: Mapping[str, Any]):
    """Plot post-disperser diffuse spectra and integrated image-plane rates."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.5), sharex=True, constrained_layout=True,
    )
    colors = {
        "camera": "tab:cyan",
        "collimator": "tab:green",
        "preoptics": "tab:blue",
        "other": "0.5",
    }
    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, spectrum in channel["spectra"].items():
            ax.plot(
                wave,
                _plot_quantity_values(spectrum),
                lw=1.0,
                color=colors.get(name, "0.5"),
                label=f"{name} diffuse",
            )
        if channel["total_spectrum"] is not None:
            ax.plot(
                wave,
                _plot_quantity_values(channel["total_spectrum"]),
                lw=1.8,
                color="black",
                alpha=0.75,
                label="total diffuse",
            )
        ax.set_title(
            f"{channel['label']} (image plane {channel['image_plane_id']}): "
            f"{channel['total_rate_ph_s_pix']:.3g} ph/s/pix",
        )
        ax.set_xlim(wave.min(), wave.max())
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spectral background [PHOTLAM equiv.]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    dedup = OrderedDict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside upper center",
        ncol=4, frameon=False,
    )
    return fig, axes


def _sum_terms(terms: Mapping[str, np.ndarray], size: int) -> np.ndarray:
    if not terms:
        return np.zeros(size, dtype=float)
    return np.sum(list(terms.values()), axis=0)


def validate_emissivity_sanity_data(data: Mapping[str, Any]) -> None:
    """Validate split emissivity sanity-check data."""
    for aperture_id, channel in data["channels"].items():
        if not channel["pre_disperser_terms"]:
            raise ValueError(
                f"No pre-disperser emissivity terms for aperture_id={aperture_id}"
            )
        if not channel["post_disperser_terms"]:
            raise ValueError(
                f"No post-disperser emissivity terms for aperture_id={aperture_id}"
            )
        checks = [
            ("pre_disperser", channel["pre_disperser_output_equiv"]),
            ("post_disperser_after_qe", channel["post_disperser_after_qe"]),
            ("detector_qe", channel["detector_qe"]),
        ]
        for label, arr in checks:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise ValueError(
                    f"Empty/non-finite {label} curve for aperture_id={aperture_id}"
                )
            if np.nanmin(finite) < -1e-6:
                warnings.warn(
                    f"Negative {label} curve for aperture_id={aperture_id}: "
                    f"{np.nanmin(finite):.3g}",
                    stacklevel=2,
                )

    if "detector_emissivity" in next(iter(data["channels"].values())):
        raise ValueError("Detector emissivity should not be part of QE validation data.")


def plot_emissivity_sanity(data: Mapping[str, Any]):
    """Plot split pre/post-disperser emissivity sanity-check data."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    group_colors = {
        "preoptics": "tab:blue",
        "collimator": "tab:green",
        "camera": "tab:cyan",
        "other": "0.5",
    }

    ymax = 0.0
    for channel in data["channels"].values():
        curves = (
            list(channel["pre_disperser_terms"].values())
            + list(channel["post_disperser_terms"].values())
            + [channel["pre_disperser_output_equiv"],
               channel["post_disperser_after_qe"]]
        )
        for curve in curves:
            finite = curve[np.isfinite(curve)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(finite)))
    ymax = max(0.05, min(1.5, ymax * 1.08))

    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, values in channel["pre_disperser_terms"].items():
            ax.plot(
                wave, values, lw=1.0, ls="--",
                color=group_colors.get(name, "0.5"),
                label=f"pre {name}",
            )
        for name, values in channel["post_disperser_terms"].items():
            ax.plot(
                wave, values, lw=1.0, ls="-",
                color=group_colors.get(name, "0.5"),
                label=f"post {name}",
            )

        ax.plot(
            wave, channel["pre_disperser_output_equiv"], lw=1.6,
            color="tab:purple", alpha=0.75, label="pre total",
        )
        ax.plot(
            wave, channel["post_disperser_after_qe"], lw=1.8,
            color="black", alpha=0.75, label="post total after QE",
        )
        ax.plot(
            wave, channel["detector_qe"], lw=0.9, ls=":",
            color="tab:red", alpha=0.8, label="QE throughput",
        )
        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dimensionless response")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    dedup = OrderedDict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside upper center",
        ncol=6, frameon=False,
    )
    return fig, axes


def detector_geometry_table(ztrain: Any) -> Table:
    """Return detector geometry/settings rows from active DetectorList effects."""
    rows: list[dict[str, Any]] = []
    for effect in active_effects(ztrain):
        if effect.__class__.__name__ != "DetectorList":
            continue
        for row in effect.table:
            rows.append({
                "detector_id": int(_row_scalar(row, "id")),
                "image_plane_id": int(effect.meta.get("image_plane_id")),
                "channel": effect_name(effect).replace("detector_", "").upper(),
                "detector": effect.meta.get("detector", ""),
                "x_size": _row_scalar(row, "x_size"),
                "y_size": _row_scalar(row, "y_size"),
                "pixel_size_mm": _row_scalar(row, "pixel_size"),
                "gain_e_per_adu": _row_scalar(row, "gain"),
            })
    return Table(rows=sorted(rows, key=lambda row: row["detector_id"]))


def _selector_value_for_detector(selector: Any, detector_row: Any) -> int | None:
    selector_key = selector.meta.get("selector_key")
    if selector_key == "detector_id":
        return int(detector_row["detector_id"])
    if selector_key == "aperture_id":
        return int(detector_row["image_plane_id"])
    return None


def _resolved_meta_summary(effect: Any, cmds: Any) -> str:
    from scopesim.utils import from_currsys

    keys = ("filename", "dit", "ndit", "value", "noise_std", "bias", "binx", "biny")
    parts = []
    for key in keys:
        if key not in effect.meta:
            continue
        value = effect.meta[key]
        try:
            value = from_currsys(value, cmds=cmds)
        except Exception:
            pass
        parts.append(f"{key}={value}")
    if not parts:
        return ""
    return ", ".join(parts)


def detector_selector_matrix(ztrain: Any) -> Table:
    """Return detector selector settings resolved for each detector row."""
    detectors = detector_geometry_table(ztrain)
    selectors = [
        effect for effect in active_effects(ztrain)
        if hasattr(effect, "wheel_effects")
    ]
    rows: list[dict[str, Any]] = []
    for detector in detectors:
        for selector in selectors:
            selector_value = _selector_value_for_detector(selector, detector)
            if selector_value is None:
                continue
            try:
                selected_effect = resolve_effect(selector, selector_value)
            except KeyError:
                continue
            rows.append({
                "detector_id": int(detector["detector_id"]),
                "channel": str(detector["channel"]),
                "detector": str(detector["detector"]),
                "selector": effect_name(selector),
                "selector_key": selector.meta.get("selector_key"),
                "selector_value": selector_value,
                "effect_class": selected_effect.__class__.__name__,
                "settings": _resolved_meta_summary(selected_effect, ztrain.cmds),
            })
    return Table(rows=rows)


def dichroic_path_throughput(
    dichroic_tree: Any,
    aperture_id: int,
    wave: u.Quantity,
) -> tuple[np.ndarray, OrderedDict[str, np.ndarray]]:
    """Return total and per-component dichroic throughput for an aperture."""
    table = dichroic_tree.table
    id_col = table.colnames[0]
    rows = table[table[id_col] == aperture_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one dichroic row for {id_col}={aperture_id}; found "
            f"{len(rows)}"
        )

    action_lookup = {"T": "transmission", "R": "reflection", "X": None}
    components: OrderedDict[str, np.ndarray] = OrderedDict()
    total = np.ones(wave.size, dtype=float)
    row = rows[0]
    for dichroic_name in table.colnames[1:]:
        action = action_lookup.get(str(row[dichroic_name]))
        if action is None:
            continue
        curve = getattr(dichroic_tree.dichroics[dichroic_name].surface, action)
        values = evaluate_curve(curve, wave)
        components[f"{dichroic_name}:{action[0].upper()}"] = values
        total *= values
    return total, components


def build_transmission_sanity_data(
    ztrain: Any,
    wave_nm: u.Quantity | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Build channel/order throughput data for transmission sanity plots."""
    wave_nm = wave_nm if wave_nm is not None else np.linspace(300, 2500, 5000) * u.nm
    wave = wave_nm.to(u.um)

    dichroic_tree = get_effect(ztrain, "dichroic_tree")
    channel_selector = get_effect(ztrain, "channel_optics_selector")
    qe_selector = get_effect(ztrain, "detector_qe_selector")
    trace_list = get_effect(ztrain, "trace_list_analytical")
    trace_eff = get_effect(ztrain, "trace_eff_analytical")
    traces_for_aperture = traces_by_aperture(trace_list)

    aperture_ids = [
        int(value)
        for value in dichroic_tree.table[dichroic_tree.table.colnames[0]]
    ]
    channels: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for aperture_id in aperture_ids:
        channel_optics = resolve_effect(channel_selector, aperture_id)
        detector_qe = resolve_effect(qe_selector, aperture_id)

        dichroic_total, dichroic_components = dichroic_path_throughput(
            dichroic_tree, aperture_id, wave,
        )
        optics_groups, group_counts = surface_list_group_throughputs(
            channel_optics, wave, groups=groups,
        )
        optics_total = (
            np.prod(list(optics_groups.values()), axis=0)
            if optics_groups
            else np.ones(wave.size)
        )
        qe_values = evaluate_throughput(detector_qe, wave)
        pre_disperser_total = dichroic_total * optics_total * qe_values

        orders: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for trace in traces_for_aperture.get(aperture_id, []):
            order_eff = _as_float_array(
                trace_eff.efficiency_generator(trace.trace_id, wave),
            )
            mask = (wave >= trace.wave_min * u.um) & (wave <= trace.wave_max * u.um)
            order_eff = np.where(mask, order_eff, np.nan)
            orders[trace.trace_id] = {
                "disperser": order_eff,
                "total": pre_disperser_total * order_eff,
                "wave_min": trace.wave_min * u.um,
                "wave_max": trace.wave_max * u.um,
            }

        channels[aperture_id] = {
            "label": channel_label(
                aperture_id, traces_for_aperture.get(aperture_id, []),
            ),
            "dichroic_total": dichroic_total,
            "dichroic_components": dichroic_components,
            "optics_groups": optics_groups,
            "optics_group_counts": group_counts,
            "optics_total": optics_total,
            "detector_qe": qe_values,
            "pre_disperser_total": pre_disperser_total,
            "orders": orders,
        }

    return {"wave_nm": wave_nm, "channels": channels}


def validate_transmission_sanity_data(data: Mapping[str, Any]) -> None:
    """Validate transmission sanity-check data."""
    for aperture_id, channel in data["channels"].items():
        if not channel["optics_groups"]:
            raise ValueError(f"No optics groups for aperture_id={aperture_id}")
        if not channel["orders"]:
            raise ValueError(f"No echelle orders for aperture_id={aperture_id}")
        arrays = [
            channel["dichroic_total"],
            channel["optics_total"],
            channel["detector_qe"],
            channel["pre_disperser_total"],
        ]
        arrays.extend(channel["optics_groups"].values())
        arrays.extend(order["disperser"] for order in channel["orders"].values())
        for arr in arrays:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                raise ValueError(
                    f"Empty/non-finite curve for aperture_id={aperture_id}"
                )
            if np.nanmin(finite) < -1e-6 or np.nanmax(finite) > 1.5:
                warnings.warn(
                    f"Throughput outside expected range for "
                    f"aperture_id={aperture_id}: "
                    f"{np.nanmin(finite):.3g}..{np.nanmax(finite):.3g}",
                    stacklevel=2,
                )


def plot_transmission_sanity(data: Mapping[str, Any]):
    """Plot channel/order throughput sanity-check data."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
        constrained_layout=True,
    )

    group_colors = {
        "preoptics": "tab:blue",
        "collimator": "tab:green",
        "camera": "tab:cyan",
        "other": "0.5",
    }

    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, values in channel["optics_groups"].items():
            ax.plot(
                wave, values, lw=1.0,
                color=group_colors.get(name, "0.5"),
                label=name,
            )

        ax.plot(
            wave, channel["dichroic_total"], lw=1.2,
            color="tab:purple", label="dichroics",
        )
        ax.plot(
            wave, channel["detector_qe"], lw=1.2,
            color="tab:red", label="detector QE",
        )

        for idx, (_trace_id, order) in enumerate(channel["orders"].items()):
            order_label = "disperser/order" if idx == 0 else None
            total_label = "total/order" if idx == 0 else None
            ax.plot(
                wave, order["disperser"], lw=0.7, color="tab:orange",
                alpha=0.35, label=order_label,
            )
            ax.plot(
                wave, order["total"], lw=1.8, color="black",
                alpha=0.65, label=total_label,
            )

        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=7, frameon=False,
    )
    return fig, axes
