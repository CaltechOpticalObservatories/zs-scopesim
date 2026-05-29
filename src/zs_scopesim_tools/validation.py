"""Validation helpers for ZShooter ScopeSim notebooks."""

from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table


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
