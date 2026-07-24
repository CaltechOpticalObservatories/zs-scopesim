"""Plotting helpers for ZShooter ScopeSim validation notebooks."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM


def _as_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def _source_label(source: Any) -> str:
    meta = getattr(source, "meta", {}) or {}
    for key in ("name", "object", "description", "function_call"):
        value = meta.get(key)
        if value:
            return str(value)
    return f"{source.__class__.__name__}@{id(source):x}"


def _row_label(source: Any, table: Table, row_index: int) -> str:
    for key in ("label", "name", "object", "source"):
        if key in table.colnames:
            value = table[key][row_index]
            if value is not None and str(value):
                return str(value)
    return f"{_source_label(source)} row {row_index}"


def _plot_spectrum_values(
    ax: Any,
    wave: u.Quantity,
    values: Any,
    *,
    label: str,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    unit = getattr(values, "unit", None)
    plot_values = _as_float_array(values)
    ylabel = "Flux density"
    if unit is not None and unit != u.dimensionless_unscaled:
        try:
            plot_values = (
                u.Quantity(values).to_value(PHOTLAM)
                * _PHOTLAM_TO_PH_S_M2_NM
            )
            ylabel = (
                r"Flux density "
                r"[photons s$^{-1}$ m$^{-2}$ nm$^{-1}$]"
            )
        except Exception:
            ylabel = f"Flux density [{unit}]"
    ax.plot(
        wave.to_value(u.um),
        plot_values,
        label=label,
        lw=linewidth,
        alpha=alpha,
        ls=linestyle,
    )
    if not ax.get_ylabel():
        ax.set_ylabel(ylabel)


def _evaluate_spectrum_values(spectrum: Any, wave: u.Quantity, weight: float = 1.0) -> Any:
    return spectrum(wave) * weight


def _source_field_image_data(field: Any) -> np.ndarray | None:
    if hasattr(field, "data"):
        return np.asarray(field.data)
    source_field = getattr(field, "field", None)
    data = getattr(source_field, "data", None)
    if data is None:
        return None
    data = np.asarray(data)
    return data if data.ndim >= 2 else None


def _is_table_source_field(field: Any) -> bool:
    table = getattr(field, "field", None)
    return isinstance(table, Table) and {"x", "y"}.issubset(table.colnames)


def _quantity_column(table: Table, name: str, unit: u.UnitBase) -> u.Quantity:
    values = table[name]
    quantity = getattr(values, "quantity", values)
    quantity = u.Quantity(quantity)
    if quantity.unit == u.dimensionless_unscaled:
        quantity = quantity.value * unit
    return quantity.to(unit)


def _plot_table_source_field(
    source: Any,
    ax_image: Any,
    ax_spectrum: Any,
    field: Any,
    wave: u.Quantity,
    *,
    individual: bool,
    spectrum_linewidth: float,
    spectrum_alpha: float,
    spectrum_linestyle: str,
) -> None:
    table = field.field
    x = _quantity_column(table, "x", u.arcsec).to_value(u.arcsec)
    y = _quantity_column(table, "y", u.arcsec).to_value(u.arcsec)
    refs = (
        np.asarray(table["ref"], dtype=int)
        if "ref" in table.colnames
        else np.zeros(len(table), dtype=int)
    )
    weights = (
        np.asarray(table["weight"], dtype=float)
        if "weight" in table.colnames
        else np.ones(len(table), dtype=float)
    )
    sizes = 36 + 84 * weights / max(np.nanmax(weights), 1.0)
    ax_image.axhline(0, color="0.65", lw=0.8, zorder=0)
    ax_image.axvline(0, color="0.65", lw=0.8, zorder=0)
    scatter = ax_image.scatter(
        x,
        y,
        c=refs,
        s=sizes,
        cmap="tab10",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    if len(table) <= 10:
        for idx, (xpos, ypos) in enumerate(zip(x, y, strict=True)):
            ax_image.annotate(
                _row_label(source, table, idx),
                (xpos, ypos),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                zorder=4,
            )
    ax_image.set_aspect("equal", adjustable="datalim")
    ax_image.set_xlabel("x [arcsec]")
    ax_image.set_ylabel("y [arcsec]")
    ax_image.set_title("Source Positions")
    if len(np.unique(refs)) > 1:
        ax_image.legend(
            *scatter.legend_elements(prop="colors", fmt="{x:.0f}"),
            title="ref",
            frameon=False,
            loc="best",
        )

    spectra = getattr(field, "spectra", {})
    if individual:
        for row_index, ref in enumerate(refs):
            spectrum = spectra.get(int(ref), spectra.get(ref))
            if spectrum is None:
                continue
            values = _evaluate_spectrum_values(
                spectrum, wave, weight=float(weights[row_index]),
            )
            _plot_spectrum_values(
                ax_spectrum,
                wave,
                values,
                label=_row_label(source, table, row_index),
                linewidth=spectrum_linewidth,
                alpha=spectrum_alpha,
                linestyle=spectrum_linestyle,
            )
        return

    total_values = None
    for row_index, ref in enumerate(refs):
        spectrum = spectra.get(int(ref), spectra.get(ref))
        if spectrum is None:
            continue
        values = _evaluate_spectrum_values(
            spectrum, wave, weight=float(weights[row_index]),
        )
        total_values = values if total_values is None else total_values + values
    if total_values is not None:
        _plot_spectrum_values(
            ax_spectrum,
            wave,
            total_values,
            label=f"{_source_label(source)} total ({len(table)} points)",
            linewidth=spectrum_linewidth,
            alpha=spectrum_alpha,
            linestyle=spectrum_linestyle,
        )


def _plot_image_source_field(
    ax_image: Any,
    ax_spectrum: Any,
    field: Any,
    wave: u.Quantity,
    *,
    spectrum_linewidth: float,
    spectrum_alpha: float,
    spectrum_linestyle: str,
) -> None:
    data = _source_field_image_data(field)
    if data is None:
        ax_image.text(
            0.5, 0.5, "No spatial image", ha="center", va="center",
            transform=ax_image.transAxes,
        )
        ax_image.set_axis_off()
    else:
        if data.ndim > 2:
            data = np.nanmean(data, axis=0)
        im = ax_image.imshow(data, origin="lower", cmap="viridis")
        ax_image.set_title("Spatial Profile")
        ax_image.set_xlabel("x pixel")
        ax_image.set_ylabel("y pixel")
        cbar = ax_image.figure.colorbar(
            im, ax=ax_image, fraction=0.046, pad=0.025,
        )
        cbar.set_label("Relative surface brightness")

    try:
        spectrum = field.spectrum
    except Exception:
        spectrum = None
    if spectrum is not None:
        _plot_spectrum_values(
            ax_spectrum,
            wave,
            _evaluate_spectrum_values(spectrum, wave),
            label="spectrum",
            linewidth=spectrum_linewidth,
            alpha=spectrum_alpha,
            linestyle=spectrum_linestyle,
        )
    else:
        for ref, candidate in getattr(field, "spectra", {}).items():
            _plot_spectrum_values(
                ax_spectrum,
                wave,
                _evaluate_spectrum_values(candidate, wave),
                label=f"ref {ref}",
                linewidth=spectrum_linewidth,
                alpha=spectrum_alpha,
                linestyle=spectrum_linestyle,
            )


def plot_source(
    source: Any,
    wave: u.Quantity | None = None,
    *,
    individual: bool = False,
    spectrum_yscale: str = "linear",
    spectrum_linewidth: float = 1.5,
    spectrum_alpha: float = 1.0,
    spectrum_linestyle: str = "-",
):
    """Plot each source field's spatial profile/positions and spectrum."""
    import matplotlib.pyplot as plt

    wave = wave if wave is not None else np.linspace(0.3, 2.5, 1001) * u.um
    num_fields = len(source.fields)
    if num_fields == 0:
        raise ValueError(
            "Source contains no fields. For point sources built with "
            "scopesim.source.source.Source, pass x=[...], y=[...], and ref=[...] "
            "arrays so ScopeSim creates a table-backed source field."
        )
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
        if _is_table_source_field(field):
            _plot_table_source_field(
                source, ax_image, ax_spectrum, field, wave,
                individual=individual,
                spectrum_linewidth=spectrum_linewidth,
                spectrum_alpha=spectrum_alpha,
                spectrum_linestyle=spectrum_linestyle,
            )
        else:
            _plot_image_source_field(
                ax_image,
                ax_spectrum,
                field,
                wave,
                spectrum_linewidth=spectrum_linewidth,
                spectrum_alpha=spectrum_alpha,
                spectrum_linestyle=spectrum_linestyle,
            )
        ax_spectrum.set_title("Spectrum")
        ax_spectrum.set_xlabel("Wavelength [um]")
        ax_spectrum.set_yscale(spectrum_yscale)
        ax_spectrum.grid(alpha=0.2)
        if ax_spectrum.get_legend_handles_labels()[0]:
            ax_spectrum.legend(frameon=False)
    return fig, axs


_PHOTLAM_TO_PH_S_M2_NM = 1e5
_SPECTRAL_SURFACE_BRIGHTNESS_LABEL = (
    r"Spectral surface brightness "
    r"[photons s$^{-1}$ m$^{-2}$ nm$^{-1}$ arcsec$^{-2}$]"
)


def _plot_spectral_surface_brightness(values: Any) -> np.ndarray:
    """Return PHOTLAM-like background density in human photon units.

    ScopeSim's spectral-background integrator treats PHOTLAM-like thermal
    spectra as per square arcsecond. The numeric conversion is therefore
    1 PHOTLAM = 1 ph s-1 cm-2 A-1 arcsec-2 = 1e5 ph s-1 m-2 nm-1 arcsec-2.
    """
    try:
        photlam_values = u.Quantity(values).to_value(PHOTLAM)
    except Exception:
        photlam_values = _as_float_array(values)
    return np.asarray(photlam_values, dtype=float) * _PHOTLAM_TO_PH_S_M2_NM


def plot_post_disperser_diffuse_background(data: Mapping[str, Any]):
    """Plot post-disperser diffuse spectra and integrated image-plane rates."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8.2), sharex=True, constrained_layout=True,
    )
    colors = {
        "camera": "tab:cyan",
        "collimator": "tab:green",
        "preoptics": "tab:blue",
        "other": "0.5",
    }
    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        trace_min = channel.get("trace_wave_min_nm", np.nan)
        trace_max = channel.get("trace_wave_max_nm", np.nan)
        if np.isfinite(trace_min) and np.isfinite(trace_max):
            ax.axvspan(
                trace_min, trace_max, color="0.2", alpha=0.08,
                label="trace wavelength span",
            )

        for name, spectrum in channel["spectra"].items():
            ax.plot(
                wave,
                _plot_spectral_surface_brightness(spectrum),
                lw=1.4,
                color=colors.get(name, "0.5"),
                label=f"{name} diffuse",
            )
        if channel.get("total_spectrum_without_blocking") is not None:
            ax.plot(
                wave,
                _plot_spectral_surface_brightness(
                    channel["total_spectrum_without_blocking"],
                ),
                lw=2.0,
                ls="--",
                color="0.25",
                alpha=0.85,
                label="total without ir block",
            )
        if channel["total_spectrum"] is not None:
            ax.plot(
                wave,
                _plot_spectral_surface_brightness(channel["total_spectrum"]),
                lw=2.2,
                alpha=0.95,
                label="total with ir block",
            )
        blocked_rate = float(channel.get("blocking_delta_rate_ph_s_pix", 0.0))
        unblocked_rate = float(
            channel.get("total_rate_without_blocking_ph_s_pix", np.nan),
        )
        if np.isfinite(unblocked_rate) and unblocked_rate > 0:
            blocked_fraction = blocked_rate / unblocked_rate
            blocked_text = (
                f"IR block removes {blocked_rate:.2g} ph/s/pix\n"
                f"({blocked_fraction:.1%} of unblocked diffuse)"
            )
        else:
            blocked_text = f"IR block removes {blocked_rate:.2g} ph/s/pix"
        ax.text(
            0.02,
            0.96,
            blocked_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "none",
                "ec": "0.75",
                "alpha": 0.9,
            },
        )
        ax.set_title(
            f"{channel['label']} Image Plane {channel['image_plane_id']}: "
            f"{channel['total_rate_ph_s_pix']:.3g} ph/s/pix",
            pad=8,
        )
        ax.set_xlim(wave.min(), wave.max())
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(_SPECTRAL_SURFACE_BRIGHTNESS_LABEL)
    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    dedup = OrderedDict(zip(labels, handles))
    fig.suptitle("Post-Disperser Diffuse Background", fontsize=12)
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside lower center",
        ncol=5, frameon=False,
    )
    return fig, axes


def plot_emissivity_sanity(data: Mapping[str, Any]):
    """Plot split pre/post-disperser emissivity sanity-check data."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    group_colors = {
        "telescope": "tab:gray",
        "preoptics": "tab:blue",
        "collimator": "tab:green",
        "camera": "tab:cyan",
        "ir_blocking_filter": "tab:olive",
        "other": "0.5",
    }

    ymax = 0.0
    ymin = np.inf
    for channel in data["channels"].values():
        curves = (
            list(channel["pre_disperser_terms"].values())
            + list(channel["post_disperser_terms"].values())
            + [channel["pre_disperser_output_equiv"],
               channel["post_disperser_after_qe"],
               channel.get("post_disperser_without_blocking_after_qe",
                           np.zeros_like(channel["post_disperser_after_qe"]))]
        )
        for curve in curves:
            curve = _plot_spectral_surface_brightness(curve)
            finite = curve[np.isfinite(curve)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(finite)))
                positive = finite[finite > 0]
                if positive.size:
                    ymin = min(ymin, float(np.nanmin(positive)))
    ymax = max(1e-12, ymax * 1.5)
    ymin = max(1e-18, ymin / 2) if np.isfinite(ymin) else 1e-18

    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, values in channel["pre_disperser_terms"].items():
            values = _plot_spectral_surface_brightness(values)
            ax.plot(
                wave, np.where(values > 0, values, np.nan),
                lw=1.0, ls="--", alpha=0.8,
                color=group_colors.get(name, "0.5"),
                label=f"pre {name}",
            )
        for name, values in channel["post_disperser_terms"].items():
            values = _plot_spectral_surface_brightness(values)
            ax.plot(
                wave, np.where(values > 0, values, np.nan),
                lw=1.1, ls="-", alpha=0.8,
                color=group_colors.get(name, "0.5"),
                label=f"post {name}",
            )

        ax.plot(
            wave,
            np.where(
                _plot_spectral_surface_brightness(
                    channel["pre_disperser_output_equiv"],
                ) > 0,
                _plot_spectral_surface_brightness(
                    channel["pre_disperser_output_equiv"],
                ),
                np.nan,
            ),
            lw=2.0,
            color="tab:purple", alpha=0.9,
            label="pre total before trace QE", zorder=8,
        )
        post_without_block = _plot_spectral_surface_brightness(
            channel.get(
                "post_disperser_without_blocking_after_qe",
                channel["post_disperser_after_qe"],
            ),
        )
        ax.plot(
            wave,
            np.where(post_without_block > 0, post_without_block, np.nan),
            lw=2.0,
            ls="--",
            color="0.35",
            alpha=0.9,
            label="post diffuse without IR block",
            zorder=9,
        )
        post_after_qe = _plot_spectral_surface_brightness(
            channel["post_disperser_after_qe"],
        )
        ax.plot(
            wave,
            np.where(post_after_qe > 0, post_after_qe, np.nan),
            lw=2.4,
            alpha=0.95,
            label="post diffuse with IR block+QE", zorder=10,
        )
        blocked_delta = channel.get("post_disperser_blocked_delta")
        if blocked_delta is not None and np.nanmax(np.abs(blocked_delta)) > 0:
            blocked_delta = _plot_spectral_surface_brightness(blocked_delta)
            ax.plot(
                wave,
                np.where(blocked_delta > 0, blocked_delta, np.nan),
                lw=1.8,
                ls="-.",
                color="tab:brown",
                alpha=0.9,
                label="removed by IR block",
                zorder=9,
            )
        peak_lines = []
        for label, values in (
            ("pre", channel["pre_disperser_output_equiv"]),
            ("post", channel["post_disperser_after_qe"]),
            ("no block", channel.get(
                "post_disperser_without_blocking_after_qe",
                channel["post_disperser_after_qe"],
            )),
        ):
            values = _plot_spectral_surface_brightness(values)
            finite = values[np.isfinite(values)]
            if finite.size:
                peak_lines.append(f"{label}: {np.nanmax(finite):.2g}")
        peak_rate = float(channel.get("post_disperser_rate_ph_s_pix", np.nan))
        if np.isfinite(peak_rate):
            peak_lines.append(f"post rate: {peak_rate:.2g} ph/s/pix")
        ax.text(
            0.02,
            0.04,
            "peak " + "\n".join(peak_lines),
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "fc": "none",
                  "ec": "0.75", "alpha": 0.9},
        )
        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(_SPECTRAL_SURFACE_BRIGHTNESS_LABEL)

    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    dedup = OrderedDict(zip(labels, handles))
    fig.legend(
        dedup.values(), dedup.keys(), loc="outside upper center",
        ncol=6, frameon=False,
    )
    return fig, axes


def plot_detector_background_budget(table: Table):
    """Plot additive detector-background and noise terms by channel."""
    import matplotlib.pyplot as plt

    channels = [str(value) for value in table["channel"]]
    x = np.arange(len(channels))
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 4.6), constrained_layout=True,
    )
    signal_ax, noise_ax = axes

    diffuse = np.asarray(table["post_diffuse_e_pix"], dtype=float)
    dark = np.asarray(table["dark_current_e_pix"], dtype=float)
    bias = np.asarray(table["bias_e_pix"], dtype=float)
    if "additive_signal_e_pix" in table.colnames:
        additive_signal = np.asarray(table["additive_signal_e_pix"], dtype=float)
    else:
        additive_signal = diffuse + dark
    signal_ax.bar(x, diffuse, width=0.7, label="post-disperser diffuse")
    signal_ax.bar(x, dark, width=0.7, bottom=diffuse, label="dark current")
    signal_ax.plot(x, bias, "o", color="black", label="bias offset")
    if "full_well_e" in table.colnames:
        full_well = np.asarray(table["full_well_e"], dtype=float)
        first_full_well = True
        for xi, well_depth in zip(x, full_well, strict=True):
            if not np.isfinite(well_depth) or well_depth <= 0:
                continue
            signal_ax.hlines(
                well_depth, xi - 0.36, xi + 0.36,
                colors="tab:red", linewidth=2.5,
                label="full well" if first_full_well else None,
            )
            first_full_well = False
    signal_ax.set_yscale("symlog", linthresh=1.0)
    signal_ax.set_xticks(x, channels)
    signal_ax.set_ylabel("Detector signal [e-/pix]")
    signal_ax.set_title("Additive Signal")
    signal_ax.grid(axis="y", alpha=0.2)
    if "saturation_status" in table.colnames:
        statuses = [str(value) for value in table["saturation_status"]]
    elif "full_well_e" in table.colnames:
        statuses = []
        full_well = np.asarray(table["full_well_e"], dtype=float)
        for signal, well_depth in zip(additive_signal, full_well, strict=True):
            if not np.isfinite(well_depth) or well_depth <= 0:
                statuses.append("unknown")
            elif signal >= well_depth:
                statuses.append("saturated")
            elif signal >= 0.8 * well_depth:
                statuses.append("near_saturation")
            else:
                statuses.append("ok")
    else:
        statuses = ["unknown"] * len(channels)
    for tick, status in zip(signal_ax.get_xticklabels(), statuses, strict=True):
        if status == "saturated":
            tick.set_color("tab:red")
            tick.set_fontweight("bold")
        elif status == "near_saturation":
            tick.set_color("tab:orange")

    if "signal_fraction_of_full_well" in table.colnames:
        well_fractions = np.asarray(table["signal_fraction_of_full_well"], dtype=float)
    elif "full_well_e" in table.colnames:
        full_well = np.asarray(table["full_well_e"], dtype=float)
        well_fractions = np.full(len(additive_signal), np.nan)
        valid = np.isfinite(full_well) & (full_well > 0)
        well_fractions[valid] = additive_signal[valid] / full_well[valid]
    else:
        well_fractions = np.full(len(additive_signal), np.nan)

    saturated = [
        channel for channel, status in zip(channels, statuses, strict=True)
        if status == "saturated"
    ]
    near_saturated = [
        channel for channel, status in zip(channels, statuses, strict=True)
        if status == "near_saturation"
    ]
    if saturated:
        saturated_fractions = np.asarray([
            fraction for fraction, status
            in zip(well_fractions, statuses, strict=True)
            if status == "saturated"
        ])
        finite_saturated = saturated_fractions[np.isfinite(saturated_fractions)]
        max_fraction = (
            np.nanmax(finite_saturated) if finite_saturated.size else np.nan
        )
        signal_ax.text(
            0.02, 0.98,
            f"SATURATED: {', '.join(saturated)} exceed full well "
            f"(max {max_fraction:.2g}x)",
            transform=signal_ax.transAxes, va="top", ha="left",
            color="tab:red", fontsize="small", fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "tab:red", "alpha": 0.85},
        )
    elif near_saturated:
        signal_ax.text(
            0.02, 0.98,
            f"Near full well: {', '.join(near_saturated)}",
            transform=signal_ax.transAxes, va="top", ha="left",
            color="tab:orange", fontsize="small",
            bbox={"facecolor": "white", "edgecolor": "tab:orange", "alpha": 0.85},
        )
    signal_ax.legend(frameon=False, fontsize="small")

    width = 0.2
    noise_terms = [
        ("diffuse shot", "diffuse_shot_noise_e_rms", "tab:blue"),
        ("dark shot", "dark_shot_noise_e_rms", "tab:green"),
        ("read", "read_noise_e_rms", "tab:orange"),
        ("total", "total_noise_e_rms", "black"),
    ]
    offsets = (np.arange(len(noise_terms)) - 1.5) * width
    for offset, (label, column, color) in zip(offsets, noise_terms, strict=True):
        noise_ax.bar(
            x + offset, np.asarray(table[column], dtype=float),
            width=width, color=color, label=label,
        )
    noise_ax.set_yscale("symlog", linthresh=1.0)
    noise_ax.set_xticks(x, channels)
    noise_ax.set_ylabel("Noise [e- RMS/pix]")
    noise_ax.set_title("Noise Terms")
    noise_ax.grid(axis="y", alpha=0.2)
    noise_ax.legend(frameon=False, fontsize="small")
    return fig, axes


def _arm_name_for_channel(label: str) -> str:
    channel_label = str(label).upper()
    if channel_label in {"B", "G", "R", "VIS"}:
        return "VIS"
    if channel_label in {"YJ", "H", "K", "NIR"}:
        return "NIR"
    raise ValueError(f"Cannot map channel label {label!r} to VIS/NIR slit arm.")


def _slit_transmission_for_channel(
    channel: Mapping[str, Any],
    slit_loss_data: Mapping[str, Any],
    wave_nm: np.ndarray,
    slit_curve_name: str,
) -> np.ndarray:
    arm_name = _arm_name_for_channel(channel["label"])
    arms = slit_loss_data["arms"]
    if arm_name not in arms:
        raise ValueError(f"slit_loss_data has no {arm_name!r} arm.")
    arm = arms[arm_name]
    if slit_curve_name not in arm["curves"]:
        raise ValueError(
            f"{arm_name} slit-loss data has no {slit_curve_name!r} curve."
        )
    curve = arm["curves"][slit_curve_name]
    arm_wave = u.Quantity(arm["wave_nm"]).to_value(u.nm)
    throughput = np.asarray(curve["throughput"], dtype=float)
    return np.interp(wave_nm, arm_wave, throughput, left=np.nan, right=np.nan)


def _order_statistic(
    channel: Mapping[str, Any],
    key: str,
    reducer: Any,
) -> np.ndarray:
    arrays = [
        np.asarray(order[key], dtype=float)
        for order in channel["orders"].values()
        if key in order
    ]
    if not arrays:
        raise ValueError(f"Channel {channel['label']} has no order {key!r} curves.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        return reducer(np.vstack(arrays), axis=0)


def _spectrograph_optics_total(channel: Mapping[str, Any]) -> np.ndarray:
    groups = [
        np.asarray(values, dtype=float)
        for name, values in channel["instrument_optics_groups"].items()
        if name != "ir_blocking_filter"
    ]
    if not groups:
        return np.ones_like(np.asarray(channel["dichroic_total"], dtype=float))
    return np.prod(groups, axis=0)


def plot_transmission_sanity(
    data: Mapping[str, Any],
    *,
    slit_loss_data: Mapping[str, Any] | None = None,
    slit_curve_name: str = "no_ao_current_adc_residual",
    summary_component_alpha: float = 0.38,
    summary_component_linewidth: float = 1.15,
):
    """Plot component-level and all-channel throughput sanity checks."""
    import matplotlib.pyplot as plt

    wave = data["wave_nm"].to_value(u.nm)
    fig = plt.figure(figsize=(16, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.35])
    component_axes = np.array([
        [fig.add_subplot(grid[row, col]) for col in range(3)]
        for row in range(2)
    ])
    summary_ax = fig.add_subplot(grid[2, :])
    axes = {"components": component_axes, "summary": summary_ax}

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    channel_colors = {
        channel["label"]: color
        for idx, channel in enumerate(data["channels"].values())
        for color in [cycle[idx % len(cycle)]]
    }

    group_colors = {
        "telescope": "tab:gray",
        "preoptics": "tab:blue",
        "collimator": "tab:green",
        "camera": "tab:cyan",
        "other": "0.5",
    }

    for ax, (aperture_id, channel) in zip(
        component_axes.flat,
        data["channels"].items(),
        strict=False,
    ):
        for name, values in channel["optics_groups"].items():
            if name == "ir_blocking_filter":
                continue
            ax.plot(
                wave,
                values,
                lw=1.0,
                alpha=0.8,
                color=group_colors.get(name, "0.5"),
                label=name,
            )
        ax.plot(
            wave,
            channel["dichroic_total"],
            lw=1.2,
            color="tab:purple",
            alpha=0.85,
            label="dichroics",
        )
        ax.plot(
            wave,
            _spectrograph_optics_total(channel),
            lw=1.0,
            ls="--",
            color="0.25",
            alpha=0.7,
            label="spectrograph optics",
        )
        for idx, order in enumerate(channel["orders"].values()):
            ax.plot(
                wave,
                order["disperser"],
                lw=1,
                color="tab:orange",
                alpha=0.32,
                label="disperser/order" if idx == 0 else None,
            )
            ax.plot(
                wave,
                order["detector_qe"],
                lw=1.1,
                color="tab:red",
                alpha=0.8,
                label="trace QE" if idx == 0 else None,
            )
            ax.plot(
                wave,
                order["instrument"],
                lw=0.85,
                color="tab:brown",
                alpha=0.8,
                label="instrument/order" if idx == 0 else None,
            )
        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

    for ax in component_axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in component_axes[:, 0]:
        ax.set_ylabel("Throughput")
    for ax in component_axes.flat[len(data["channels"]):]:
        ax.axis("off")

    component_legend: OrderedDict[str, Any] = OrderedDict()
    for ax in component_axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            if label and not label.startswith("_"):
                component_legend.setdefault(label, handle)
    fig.legend(
        list(component_legend.values()),
        list(component_legend),
        loc="outside upper center",
        ncol=7,
        frameon=False,
        fontsize="small",
    )

    summary_component_label_used: set[str] = set()
    for channel in data["channels"].values():
        label = channel["label"]
        color = channel_colors[label]
        if slit_loss_data is not None:
            slit_transmission = _slit_transmission_for_channel(channel, slit_loss_data, wave, slit_curve_name)
        else:
            slit_transmission = np.ones_like(wave)
        summary_components = (
            ("telescope", channel["telescope_throughput"], ":", 1.1),
            ("dichroic", channel["dichroic_total"], "-.",summary_component_linewidth),
            ("spectrograph optics", _spectrograph_optics_total(channel), (0, (5, 2)),summary_component_linewidth),
            ("active slit", slit_transmission, (0, (1, 2)), 1.1),
            ("trace QE median", _order_statistic(channel, "detector_qe", np.nanmedian), (0, (3, 1, 1, 1)),summary_component_linewidth),
        )
        for component_name, values, linestyle, linewidth in summary_components:
            summary_ax.plot(
                wave,
                values,
                lw=linewidth,
                ls=linestyle,
                color=color,
                alpha=summary_component_alpha,
                label=(
                    component_name
                    if component_name not in summary_component_label_used
                    else None
                ),
            )
            summary_component_label_used.add(component_name)

        for idx, order in enumerate(channel["orders"].values()):
            instrument_label = f"{label} instrument" if idx == 0 else None
            total_label = f"{label} total" if idx == 0 else None
            summary_ax.plot(
                wave,
                order["detector_qe"],
                lw=0.55,
                ls=(0, (3, 1, 1, 1)),
                color=color,
                alpha=0.22,
                label=None,
            )
            summary_ax.plot(
                wave,
                order["instrument"],
                lw=0.95,
                ls="--",
                color=color,
                alpha=0.48,
                label=instrument_label,
            )
            summary_ax.plot(
                wave,
                order["total_with_telescope_no_slit"] * slit_transmission,
                lw=1.55,
                color=color,
                alpha=0.86,
                label=total_label,
            )

    summary_ax.set_title("All Channels: Components, Instrument, and Total")
    summary_ax.set_xlim(wave.min(), wave.max())
    summary_ax.set_ylim(0, 1.05)
    summary_ax.set_xlabel("Wavelength [nm]")
    summary_ax.set_ylabel("Throughput")
    summary_ax.grid(alpha=0.2)
    summary_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=7,
        frameon=False,
        fontsize="small",
    )
    return fig, axes


def plot_slit_pair_geometry(
    table: Table,
    *,
    slit_width: u.Quantity = 0.7 * u.arcsec,
    slit_length: u.Quantity = 10 * u.arcsec,
):
    """Plot a two-point source geometry relative to a rectangular slit."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x = u.Quantity(table["x"]).to_value(u.arcsec)
    y = u.Quantity(table["y"]).to_value(u.arcsec)
    width = u.Quantity(slit_width).to_value(u.arcsec)
    length = u.Quantity(slit_length).to_value(u.arcsec)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.add_patch(Rectangle(
        (-0.5 * length, -0.5 * width),
        length,
        width,
        fill=False,
        lw=1.6,
        color="black",
        label="slit",
    ))
    ax.scatter(x, y, s=50, color="tab:red", zorder=3, label="sources")
    for idx, (xpos, ypos) in enumerate(zip(x, y, strict=True)):
        ax.annotate(str(idx), (xpos, ypos), xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    pad = max(width, np.ptp(y) if y.size > 1 else width, 0.25) * 0.7
    ax.set_xlim(min(-0.5 * length, x.min()) - pad,
                max(0.5 * length, x.max()) + pad)
    ax.set_ylim(min(-width, y.min()) - pad, max(width, y.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Along slit [arcsec]")
    ax.set_ylabel("Across slit [arcsec]")
    ax.legend(frameon=False)
    return fig, ax


def plot_slit_adc_psf_scenes(data: Mapping[str, Any]):
    """Plot PSF-convolved slit scenes with AD-only and ADC-residual shifts."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    scenarios = data["scenarios"]
    variants = data["variants"]
    x = u.Quantity(data["x_arcsec"]).to_value(u.arcsec)
    y = u.Quantity(data["y_arcsec"]).to_value(u.arcsec)
    extent = [x.min(), x.max(), y.min(), y.max()]
    width = u.Quantity(data["slit_width_arcsec"]).to_value(u.arcsec)
    length = u.Quantity(data["slit_length_arcsec"]).to_value(u.arcsec)

    nrows = len(scenarios)
    ncols = len(variants)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 4.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (scenario_name, scenario) in enumerate(scenarios.items()):
        table = scenario["positions"]
        xpos = u.Quantity(table["x"]).to_value(u.arcsec)
        ypos = u.Quantity(table["y"]).to_value(u.arcsec)
        for col, (variant_name, variant) in enumerate(variants.items()):
            ax = axes[row, col]
            image = scenario["images"][variant_name]
            finite = image[np.isfinite(image)]
            if finite.size:
                vmax = np.nanpercentile(finite, 99.6)
                vmin = np.nanpercentile(finite, 5.0)
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.add_patch(Rectangle(
                (-0.5 * length, -0.5 * width),
                length,
                width,
                fill=False,
                lw=2.4,
                edgecolor="white",
                label="slit" if row == 0 and col == 0 else None,
            ))
            ax.scatter(
                xpos,
                ypos,
                s=54,
                marker="o",
                facecolors="none",
                edgecolors="#7FDBFF",
                linewidths=1.8,
                label="source centers" if row == 0 and col == 0 else None,
            )
            ax.axvline(0, color="white", lw=0.8, alpha=0.45)
            ax.axhline(0, color="white", lw=0.8, alpha=0.25)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{scenario_name}\n{variant['label']}", fontsize=10)
            if row == nrows - 1:
                ax.set_xlabel("Along slit [arcsec]")
            if col == 0:
                ax.set_ylabel("Across slit [arcsec]")
            if col == ncols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
                cbar.set_label("Relative PSF intensity")

    fig.suptitle(
        "Slit/AD/PSF Scene Check "
        f"(airmass {data['airmass']:.2f}, "
        f"seeing {data['seeing_arcsec'].to_value(u.arcsec):.2f} arcsec)",
        fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="outside lower center", ncol=len(handles),
            frameon=False,
        )
    return fig, axes


def _slit_loss_curve_psf_key(curve: Mapping[str, Any]) -> str:
    if "psf_mode" in curve:
        return str(curve["psf_mode"])
    label = str(curve.get("label", "curve"))
    return label.split(",", 1)[0]


def _slit_loss_curve_slit_width(
    curve: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> float:
    width = curve.get("slit_width_arcsec", arm.get("slit_width_arcsec"))
    return float(u.Quantity(width).to_value(u.arcsec))


def _slit_loss_curve_airmass_key(curve: Mapping[str, Any]) -> float | None:
    if "airmass" not in curve:
        return None
    return round(float(curve["airmass"]), 6)


def _slit_loss_curve_slit_role(curve: Mapping[str, Any]) -> str:
    return str(curve.get("slit_role", "current"))


def _slit_loss_curve_adc_state(curve: Mapping[str, Any]) -> str:
    return str(curve.get("adc_state", "adc_residual"))


def _slit_loss_airmass_alpha(curve: Mapping[str, Any]) -> float:
    if "airmass" not in curve:
        return 0.95
    if curve.get("current_airmass", False):
        return 0.98
    if float(curve["airmass"]) <= 1.000001:
        return 0.34
    return 0.56


def _format_slit_value(value: float) -> str:
    return f'{value:.2f}"'


def _format_slit_values_by_arm(values: Mapping[str, float]) -> str:
    finite = [
        float(value)
        for value in values.values()
        if np.isfinite(value)
    ]
    if not finite:
        return "--"
    if all(np.isclose(value, finite[0], rtol=0.0, atol=1.0e-6) for value in finite):
        return _format_slit_value(finite[0])
    return "/".join(_format_slit_value(value) for value in finite)


def _arm_marker_map(arms: Mapping[str, Any]) -> dict[str, str]:
    preferred = {"VIS": "*", "NIR": "^"}
    fallback = ["*", "^", "+", "x"]
    markers = {}
    for idx, arm_name in enumerate(arms):
        markers[arm_name] = preferred.get(
            str(arm_name).upper(),
            fallback[idx % len(fallback)],
        )
    return markers


def _slit_role_for_arm(arm: Mapping[str, Any]) -> str:
    current = float(u.Quantity(arm["slit_width_arcsec"]).to_value(u.arcsec))
    if "selector_slit_widths_arcsec" not in arm:
        return "current"
    selector = u.Quantity(
        arm["selector_slit_widths_arcsec"],
    ).to_value(u.arcsec)
    selector = np.asarray(selector, dtype=float)
    selector = selector[np.isfinite(selector)]
    if selector.size < 2:
        return "current"
    if np.isclose(current, np.nanmin(selector), rtol=0.0, atol=1.0e-6):
        return "narrowest"
    if np.isclose(current, np.nanmax(selector), rtol=0.0, atol=1.0e-6):
        return "widest"
    return "current"


def _selected_slit_values_by_arm(arms: Mapping[str, Any]) -> dict[str, float]:
    return {
        arm_name: float(u.Quantity(arm["slit_width_arcsec"]).to_value(u.arcsec))
        for arm_name, arm in arms.items()
        if "slit_width_arcsec" in arm
    }


def _slit_role_values_by_arm(
    arms: Mapping[str, Any],
    visible_by_arm: Mapping[str, list[Mapping[str, Any]]],
) -> OrderedDict[str, dict[str, float]]:
    role_values: OrderedDict[str, dict[str, float]] = OrderedDict()
    for role in ("narrowest", "current", "widest"):
        role_values[role] = {}
    for arm_name, arm in arms.items():
        for curve in visible_by_arm.get(arm_name, []):
            role = _slit_loss_curve_slit_role(curve)
            role_values.setdefault(role, {})
            role_values[role].setdefault(
                arm_name,
                _slit_loss_curve_slit_width(curve, arm),
            )
    return OrderedDict(
        (role, values) for role, values in role_values.items() if values
    )


def _slit_legend_label(
    role: str,
    values_by_arm: Mapping[str, float],
    selected_values: Mapping[str, float],
    selected_roles: Mapping[str, str],
    markers: Mapping[str, str],
) -> tuple[str, set[str]]:
    if role == "current":
        selected_current = {
            arm: value
            for arm, value in selected_values.items()
            if selected_roles.get(arm) == "current"
        }
        if selected_current:
            value_label = _format_slit_values_by_arm(selected_current)
            if len(selected_current) > 1:
                finite = list(selected_current.values())
                same = all(
                    np.isclose(value, finite[0], rtol=0.0, atol=1.0e-6)
                    for value in finite
                )
                suffix = "Selected slit" if same else "Current slit"
                return f"{value_label} {suffix}", set()
            arm_name = next(iter(selected_current))
            marker = markers.get(arm_name, "")
            return f"{value_label}{marker} Current slit", {marker} if marker else set()
        return _format_slit_values_by_arm(values_by_arm), set()

    value_label = _format_slit_values_by_arm(values_by_arm)
    current_markers = {
        markers[arm_name]
        for arm_name in values_by_arm
        if selected_roles.get(arm_name) == role
        and arm_name in markers
    }
    marker_label = "".join(
        markers[arm_name]
        for arm_name in values_by_arm
        if selected_roles.get(arm_name) == role
        and arm_name in markers
    )
    return f"{value_label}{marker_label}", current_markers


def _add_adc_off_ticks(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: Any,
    alpha: float,
    linewidth: float,
    zorder: float,
    tick_count: int = 13,
) -> None:
    from matplotlib.collections import LineCollection

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if finite.size < 3:
        return
    candidates = finite[1:-1]
    if candidates.size == 0:
        return
    count = min(tick_count, candidates.size)
    indices = np.unique(
        np.linspace(0, candidates.size - 1, count).round().astype(int),
    )
    tick_indices = candidates[indices]

    x_min = float(np.nanmin(x[finite]))
    x_span = float(np.nanmax(x[finite]) - x_min)
    if not np.isfinite(x_span) or x_span <= 0:
        return
    y_min, y_span = 0.0, 1.0
    half_len = 0.009
    segments = []
    for idx in tick_indices:
        left = max(idx - 1, 0)
        right = min(idx + 1, x.size - 1)
        if not np.isfinite(y[left]) or not np.isfinite(y[right]):
            continue
        dx = (x[right] - x[left]) / x_span
        dy = (y[right] - y[left]) / y_span
        norm = np.hypot(dx, dy)
        if not np.isfinite(norm) or norm <= 0:
            continue
        nx, ny = -dy / norm, dx / norm
        x0 = (x[idx] - x_min) / x_span
        y0 = (y[idx] - y_min) / y_span
        p0 = (
            x_min + (x0 - half_len * nx) * x_span,
            y_min + (y0 - half_len * ny) * y_span,
        )
        p1 = (
            x_min + (x0 + half_len * nx) * x_span,
            y_min + (y0 + half_len * ny) * y_span,
        )
        segments.append([p0, p1])
    if not segments:
        return
    ax.add_collection(LineCollection(
        segments,
        colors=[color],
        linewidths=max(0.75, linewidth * 0.55),
        alpha=alpha,
        zorder=zorder + 0.2,
        capstyle="round",
    ))


def _visible_slit_loss_curves(
    arm: Mapping[str, Any],
    *,
    show_airmass_states: bool,
    show_adc_states: bool,
) -> list[Mapping[str, Any]]:
    curves = []
    for curve in arm["curves"].values():
        if curve.get("alias_for"):
            continue
        if (
            not show_airmass_states
            and "current_airmass" in curve
            and not curve["current_airmass"]
        ):
            continue
        if (
            not show_adc_states
            and "current_adc" in curve
            and not curve["current_adc"]
        ):
            continue
        curves.append(curve)
    return curves


def plot_slit_loss_by_arm(
    data: Mapping[str, Any],
    *,
    show_airmass_states: bool = False,
    show_adc_states: bool = False,
):
    """Plot centered point-source slit loss for each spectrograph arm."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arms = data["arms"]
    visible_by_arm = OrderedDict(
        (
            arm_name,
            _visible_slit_loss_curves(
                arm,
                show_airmass_states=show_airmass_states,
                show_adc_states=show_adc_states,
            ),
        )
        for arm_name, arm in arms.items()
    )
    all_visible = [
        curve
        for curves in visible_by_arm.values()
        for curve in curves
    ]
    if not all_visible:
        raise ValueError("No slit-loss curves remain after plot filters.")

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    psf_order = list(data.get("psf_modes", {}))
    for curve in all_visible:
        key = _slit_loss_curve_psf_key(curve)
        if key not in psf_order:
            psf_order.append(key)
    psf_colors = {
        key: cycle[idx % len(cycle)]
        for idx, key in enumerate(psf_order)
    }

    airmass_values = sorted({
        value
        for curve in all_visible
        for value in [_slit_loss_curve_airmass_key(curve)]
        if value is not None
    })
    slit_styles = {
        "narrowest": (0, (1, 1.6)),
        "current": "-",
        "widest": (0, (6, 2.4)),
    }
    selected_slits = _selected_slit_values_by_arm(arms)
    selected_roles = {
        arm_name: _slit_role_for_arm(arm)
        for arm_name, arm in arms.items()
    }
    arm_markers = _arm_marker_map(arms)
    slit_role_values = _slit_role_values_by_arm(arms, visible_by_arm)
    fig, axes = plt.subplots(
        1,
        len(arms),
        figsize=(6.4 * len(arms), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for idx, (ax, (arm_name, arm)) in enumerate(
        zip(axes.flat, arms.items(), strict=True),
    ):
        wave = u.Quantity(arm["wave_nm"]).to_value(u.nm)
        curves = sorted(
            visible_by_arm[arm_name],
            key=lambda curve: (
                bool(curve.get("current_slit", False)),
                bool(curve.get("current_airmass", False)),
                bool(curve.get("current_adc", False)),
            ),
        )
        for curve in curves:
            psf_key = _slit_loss_curve_psf_key(curve)
            slit_role = _slit_loss_curve_slit_role(curve)
            adc_state = _slit_loss_curve_adc_state(curve)
            current_curve = (
                bool(curve.get("current_slit", False))
                and bool(curve.get("current_airmass", False))
                and bool(curve.get("current_adc", False))
            )
            linewidth = 2.7 if current_curve else 1.25
            alpha = _slit_loss_airmass_alpha(curve)
            zorder = 4.5 if current_curve else 2.0
            color = psf_colors.get(psf_key, curve.get("color"))
            line, = ax.plot(
                wave,
                curve["loss"],
                lw=linewidth,
                color=color,
                ls=slit_styles.get(slit_role, "-"),
                alpha=alpha,
                label="_nolegend_",
                zorder=zorder,
            )
            if adc_state == "ad_only":
                _add_adc_off_ticks(
                    ax,
                    wave,
                    np.asarray(curve["loss"], dtype=float),
                    color=line.get_color(),
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=zorder,
                )
        title_bits = [f"{arm_name} slit loss"]
        if "slit_width_arcsec" in arm:
            active_slit = u.Quantity(arm["slit_width_arcsec"]).to_value(u.arcsec)
            title_bits.append(
                f'active slit {active_slit:.2f}"'
            )
        ax.set_title(
            "\n".join(title_bits),
            pad=8,
        )
        ax.set_xlabel("Wavelength [nm]")
        if idx == 0:
            ax.set_ylabel("Slit loss fraction")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    def legend_heading(label: str):
        return Line2D([0], [0], color="none", lw=0, label=label)

    def legend_blank():
        return Line2D([0], [0], color="none", lw=0, label=" ")

    seeing_handles = [legend_heading("Seeing")]
    for key in psf_order:
        matches = [
            curve
            for curve in all_visible
            if _slit_loss_curve_psf_key(curve) == key
        ]
        if not matches:
            continue
        mode_meta = data.get("psf_modes", {}).get(key, {})
        label = str(mode_meta.get("label", matches[0].get("psf_label", key)))
        current = bool(
            mode_meta.get(
                "current",
                any(curve.get("current_psf", False) for curve in matches),
            )
        )
        if current:
            label = f"{label} current"
        seeing_handles.append(
            Line2D(
                [0],
                [0],
                color=psf_colors[key],
                lw=2.4,
                label=label,
            )
        )

    slit_handles = [legend_heading("Slit")]
    slit_current_markers: set[str] = set()
    for role, values_by_arm in slit_role_values.items():
        label, markers_used = _slit_legend_label(
            role,
            values_by_arm,
            selected_slits,
            selected_roles,
            arm_markers,
        )
        slit_current_markers.update(markers_used)
        role_is_selected = any(
            selected_roles.get(arm_name) == role
            for arm_name in values_by_arm
        )
        slit_handles.append(
            Line2D(
                [0],
                [0],
                color=cycle[0],
                lw=2.7 if role_is_selected else 1.5,
                ls=slit_styles.get(role, "-"),
                label=label,
            )
        )
    if slit_current_markers:
        ordered_markers = [
            arm_markers[arm_name]
            for arm_name in arms
            if arm_markers.get(arm_name) in slit_current_markers
        ]
        if len(ordered_markers) > 1:
            marker_note = f"{'/'.join(ordered_markers)} Current slit"
        else:
            marker = ordered_markers[0]
            arm_name = next(
                name for name, value in arm_markers.items()
                if value == marker
            )
            marker_note = f"{marker} {arm_name} current slit"
        slit_handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                lw=0,
                label=marker_note,
            )
        )

    airmass_handles = []
    if airmass_values:
        airmass_handles.append(legend_heading("Airmass"))
        for value in airmass_values:
            matches = [
                curve
                for curve in all_visible
                if np.isclose(_slit_loss_curve_airmass_key(curve), value)
            ]
            label = str(matches[0].get("airmass_label", f"X={value:.2f}"))
            if any(curve.get("current_airmass", False) for curve in matches):
                label = f"{label} current"
            airmass_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cycle[0],
                    lw=2.2,
                    alpha=_slit_loss_airmass_alpha(matches[0]),
                    label=label,
                )
            )

    adc_states = []
    for curve in all_visible:
        state = _slit_loss_curve_adc_state(curve)
        if state and state not in adc_states:
            adc_states.append(state)
    adc_order = [state for state in ("adc_residual", "ad_only") if state in adc_states]
    adc_order.extend(state for state in adc_states if state not in adc_order)
    adc_handles = []
    if adc_order:
        adc_handles.append(legend_heading("ADC"))
    for state in adc_order:
        matches = [
            curve
            for curve in all_visible
            if _slit_loss_curve_adc_state(curve) == state
        ]
        label = "on" if state == "adc_residual" else "off"
        if state not in {"adc_residual", "ad_only"}:
            label = str(matches[0].get("adc_label", state))
        marker = "|" if state == "ad_only" else None
        adc_handles.append(
            Line2D(
                [0],
                [0],
                color=cycle[0],
                lw=2.2,
                marker=marker,
                markersize=8 if marker else 0,
                markeredgewidth=1.1,
                label=label,
            )
        )

    legend_columns = [
        column
        for column in (seeing_handles, slit_handles, airmass_handles, adc_handles)
        if column
    ]
    n_legend_columns = len(legend_columns)
    n_legend_rows = max(len(column) for column in legend_columns)
    handles = []
    for row in range(n_legend_rows):
        for column in legend_columns:
            handles.append(column[row] if row < len(column) else legend_blank())
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="outside lower center",
        ncol=max(1, n_legend_columns),
        fontsize="small",
    )
    return fig, axes


def plot_slit_width_loss(data: Mapping[str, Any]):
    """Plot centered point-source slit loss as a function of slit width."""
    import matplotlib.pyplot as plt

    arms = data["arms"]
    if len(arms) != 2:
        raise ValueError(
            "plot_slit_width_loss expects the matched two-arm data produced by "
            "build_slit_width_loss_data."
        )
    arm_names = list(arms)
    arm_waves = {
        name: u.Quantity(arms[name]["wavelengths_nm"]).to_value(u.nm)
        for name in arm_names
    }
    n_waves = len(arm_waves[arm_names[0]])
    if any(len(values) != n_waves for values in arm_waves.values()):
        raise ValueError("Matched slit-width loss arms have different wavelengths.")

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, n_waves))
    curve_labels: dict[tuple[str, str, float], str] = {}
    curve_colors: dict[tuple[str, float], Any] = {}
    psf_modes = data["psf_modes"]
    for color, wave_pair in zip(
        colors,
        zip(*(arm_waves[name] for name in arm_names), strict=True),
        strict=True,
    ):
        wave_text = "/".join(f"{wave_nm:.0f}" for wave_nm in wave_pair)
        for arm_name, wave_nm in zip(arm_names, wave_pair, strict=True):
            wave_key = round(float(wave_nm), 9)
            curve_colors[(arm_name, wave_key)] = color
            for psf_mode, psf_spec in psf_modes.items():
                curve_labels[(arm_name, psf_mode, wave_key)] = (
                    f"{psf_spec['label']}, {wave_text} nm"
                )

    fig, axes = plt.subplots(
        1,
        len(arms),
        figsize=(6.4 * len(arms), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (arm_name, arm) in zip(axes.flat, arms.items(), strict=True):
        slit_widths = u.Quantity(arm["slit_widths_arcsec"]).to_value(u.arcsec)
        for curve in arm["curves"].values():
            wave_key = round(float(curve["wavelength_nm"]), 9)
            color = curve_colors[(arm_name, wave_key)]
            ax.plot(
                slit_widths,
                curve["loss"],
                lw=2.4,
                color=color,
                ls=curve.get("linestyle", "-"),
                label=curve_labels[(arm_name, curve["psf_mode"], wave_key)],
            )
            selector_slits = u.Quantity(
                arm.get("selector_slit_widths_arcsec", []),
            ).to_value(u.arcsec)
            selector_slits = selector_slits[
                (selector_slits >= slit_widths.min())
                & (selector_slits <= slit_widths.max())
            ]
            if selector_slits.size:
                ax.scatter(
                    selector_slits,
                    np.interp(selector_slits, slit_widths, curve["loss"]),
                    s=34,
                    marker="o",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=4,
                    label="available slit widths",
                )

        current_slit = u.Quantity(
            arm["current_slit_width_arcsec"],
        ).to_value(u.arcsec)
        if np.isfinite(current_slit):
            ax.axvline(
                current_slit, color="0.25", lw=1.6, ls=":",
                label="current slit",
            )
        ax.set_title(f"{arm_name} PSF Loss vs Slit Width", pad=8)
        ax.set_xlabel("Slit width [arcsec]")
        ax.set_ylabel("Slit loss fraction")
        ax.set_ylim(0, 1)
        ax.set_xlim(slit_widths.min(), slit_widths.max())
        ax.grid(alpha=0.25)

    legend_entries: OrderedDict[str, Any] = OrderedDict()
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_entries.setdefault(label, handle)
    fig.legend(
        list(legend_entries.values()), list(legend_entries),
        loc="outside lower center",
        ncol=min(4, max(1, len(legend_entries))),
        frameon=False,
        fontsize="small",
    )
    return fig, axes


def _readout_image_data(channel_hdul: Any) -> np.ndarray:
    if isinstance(channel_hdul, np.ndarray):
        return np.asarray(channel_hdul, dtype=float)
    return np.asarray(_readout_image_hdu(channel_hdul).data, dtype=float)


def _readout_image_hdu(channel_hdul: Any) -> Any:
    if isinstance(channel_hdul, np.ndarray):
        from astropy.io import fits

        return fits.ImageHDU(data=np.asarray(channel_hdul, dtype=float))
    image_hdu = (
        channel_hdul
        if hasattr(channel_hdul, "data")
        else channel_hdul[1]
    )
    return image_hdu


def _write_readout_product(path: Path, channel_hdul: Any) -> None:
    if isinstance(channel_hdul, np.ndarray):
        from astropy.io import fits

        fits.PrimaryHDU(data=np.asarray(channel_hdul, dtype=float)).writeto(
            path,
            overwrite=True,
        )
        return

    if hasattr(channel_hdul, "writeto"):
        channel_hdul.writeto(path, overwrite=True)
        return

    image_hdu = _readout_image_hdu(channel_hdul)
    if hasattr(image_hdu, "writeto"):
        image_hdu.writeto(path, overwrite=True)
        return

    from astropy.io import fits

    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=_readout_image_data(channel_hdul)),
    ]).writeto(path, overwrite=True)


def _readout_delta_images(signal_hdul: Any, reference_hdul: Any) -> list[np.ndarray]:
    signal_readouts = list(signal_hdul)
    reference_readouts = list(reference_hdul)
    if len(signal_readouts) != len(reference_readouts):
        raise ValueError(
            "signal_hdul and reference_hdul contain different readout counts: "
            f"{len(signal_readouts)} != {len(reference_readouts)}"
        )
    return [
        _readout_image_data(signal) - _readout_image_data(reference)
        for signal, reference in zip(signal_readouts, reference_readouts, strict=True)
    ]


def _clip_fraction_label(clip: float) -> str:
    return f"p{100.0 * clip:.4g}"


def _readout_display_limits(
    data: np.ndarray,
    clip: float | None,
    *,
    symmetric: bool,
    zero_floor: bool = False,
    vmin: Any = None,
    vmax: Any = None,
) -> tuple[float, float, str]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (-1.0, 1.0, "no finite data") if symmetric else (0.0, 1.0, "no finite data")

    explicit_vmin = vmin is not None
    explicit_vmax = vmax is not None
    if explicit_vmin:
        vmin = float(vmin)
    if explicit_vmax:
        vmax = float(vmax)

    if clip is None:
        if symmetric:
            limit = float(np.nanmax(np.abs(finite)))
            if not np.isfinite(limit) or limit <= 0:
                limit = 1.0
            vmin = -limit if vmin is None else vmin
            vmax = limit if vmax is None else vmax
            label = "limits" if explicit_vmin or explicit_vmax else "unclipped"
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                raise ValueError(f"Invalid display limits: vmin={vmin}, vmax={vmax}")
            return vmin, vmax, label
        if vmin is None:
            vmin = float(np.nanmin(finite))
            if zero_floor:
                vmin = 0.0
        if vmax is None:
            vmax = float(np.nanmax(finite))
        label = "limits" if explicit_vmin or explicit_vmax else "unclipped"
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            if explicit_vmin or explicit_vmax:
                raise ValueError(f"Invalid display limits: vmin={vmin}, vmax={vmax}")
            vmax = vmin + 1.0
        return vmin, vmax, label

    clip = float(clip)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be None or a positive value.")

    if clip <= 1:
        if symmetric:
            limit = float(np.nanquantile(np.abs(finite), clip))
            label = f"{_clip_fraction_label(clip)} |value|"
        else:
            if vmin is None:
                vmin = 0.0 if zero_floor else float(np.nanmin(finite))
            if vmax is None:
                vmax = float(np.nanquantile(finite, clip))
            label = _clip_fraction_label(clip)
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                if explicit_vmin or explicit_vmax:
                    raise ValueError(f"Invalid display limits: vmin={vmin}, vmax={vmax}")
                vmax = vmin + 1.0
            return vmin, vmax, label
    else:
        limit = clip
        label = f"{clip:.4g}"
        if not symmetric:
            if vmin is None:
                vmin = 0.0 if zero_floor else min(float(np.nanmin(finite)), 0.0)
            if vmax is None:
                vmax = clip
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                if explicit_vmin or explicit_vmax:
                    raise ValueError(f"Invalid display limits: vmin={vmin}, vmax={vmax}")
                vmax = vmin + 1.0
            return vmin, vmax, label

    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    vmin = -limit if vmin is None else vmin
    vmax = limit if vmax is None else vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid display limits: vmin={vmin}, vmax={vmax}")
    return vmin, vmax, label


def _is_scalar_limit(value: Any) -> bool:
    return value is None or isinstance(value, str) or np.ndim(value) == 0


def _readout_panel_limits(
    value: Any,
    titles: Sequence[str],
    n_images: int,
    name: str,
) -> list[float | None]:
    if isinstance(value, Mapping):
        panel_values: list[float | None] = []
        for idx, title in enumerate(titles):
            item = value.get(title, value.get(idx, None))
            panel_values.append(None if item is None else float(item))
        return panel_values

    if _is_scalar_limit(value):
        return [None if value is None else float(value)] * n_images

    values = list(value)
    if len(values) != n_images:
        raise ValueError(
            f"{name} sequence must have one value per image: "
            f"{len(values)} != {n_images}"
        )
    return [None if item is None else float(item) for item in values]


def _readout_group_limit_value(
    values: Sequence[float | None],
    group: Sequence[int],
    reducer: Any,
) -> float | None:
    explicit = [values[idx] for idx in group if values[idx] is not None]
    if not explicit:
        return None
    return float(reducer(explicit))


def _readout_image_norm(
    data: np.ndarray,
    *,
    image_scale: str,
    vmin: float,
    vmax: float,
):
    from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm

    scale = image_scale.lower()
    if scale == "linear":
        return Normalize(vmin=vmin, vmax=vmax)
    if scale == "log":
        finite_positive = data[np.isfinite(data) & (data > 0)]
        if finite_positive.size == 0:
            raise ValueError("image_scale='log' requires positive finite image data.")
        positive_floor = float(np.nanmin(finite_positive))
        return LogNorm(vmin=max(vmin, positive_floor), vmax=vmax)
    if scale == "sqrt":
        if vmin < 0:
            raise ValueError(
                "image_scale='sqrt' requires a non-negative display range; "
                "use zero_floor=True or image_scale='symlog' for signed data."
            )
        return PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)
    if scale == "symlog":
        return SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax)
    raise ValueError("image_scale must be 'linear', 'log', 'sqrt', or 'symlog'.")


def _view_value(view: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if view is None:
        return default
    for key in keys:
        if key in view:
            return view[key]
    return default


def _detector_grid_axes(n_images: int, *, row_height: float = 3.6):
    import matplotlib.pyplot as plt

    ncols = min(3, max(1, n_images))
    nrows = int(np.ceil(n_images / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, row_height * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def _center_detector_axes(axes: Any, n_images: int) -> None:
    for ax in axes.flat[:n_images]:
        ax.set_anchor("C")


def _readout_scale_groups(
    shared_scale: bool | Sequence[Sequence[int | str]],
    titles: Sequence[str],
    n_images: int,
) -> list[list[int]]:
    if shared_scale is False:
        return [[idx] for idx in range(n_images)]
    if shared_scale is True:
        return [list(range(n_images))]
    title_to_index = {title: idx for idx, title in enumerate(titles)}
    groups: list[list[int]] = []
    seen: set[int] = set()
    for group in shared_scale:
        indices: list[int] = []
        for item in group:
            idx = title_to_index[item] if isinstance(item, str) else int(item)
            if idx < 0 or idx >= n_images:
                raise IndexError(f"Readout scale index out of range: {idx}")
            indices.append(idx)
            seen.add(idx)
        if indices:
            groups.append(indices)
    for idx in range(n_images):
        if idx not in seen:
            groups.append([idx])
    return groups


def _readout_group_limits(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    groups: Sequence[Sequence[int]],
    *,
    clip: float | None,
    symmetric: bool,
    zero_floor: bool,
    vmin: Any,
    vmax: Any,
) -> dict[int, tuple[float, float, str, bool]]:
    panel_vmins = _readout_panel_limits(vmin, titles, len(images), "vmin")
    panel_vmaxs = _readout_panel_limits(vmax, titles, len(images), "vmax")
    limits: dict[int, tuple[float, float, str, bool]] = {}
    for group in groups:
        finite_values = [
            np.asarray(images[idx])[np.isfinite(images[idx])]
            for idx in group
        ]
        finite_values = [values for values in finite_values if values.size]
        if finite_values:
            combined = np.concatenate(finite_values)
        else:
            combined = np.array([], dtype=float)
        group_vmin_arg = _readout_group_limit_value(panel_vmins, group, min)
        group_vmax_arg = _readout_group_limit_value(panel_vmaxs, group, max)
        group_vmin, group_vmax, clip_label = _readout_display_limits(
            combined,
            clip,
            symmetric=symmetric,
            zero_floor=zero_floor,
            vmin=group_vmin_arg,
            vmax=group_vmax_arg,
        )
        is_shared = len(group) > 1
        for idx in group:
            limits[idx] = (group_vmin, group_vmax, clip_label, is_shared)
    return limits


def _plot_detector_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    *,
    clip: float | None,
    symmetric: bool,
    cmap: str,
    title_suffix: str = "",
    annotate_delta: bool = False,
    annotate_stats: bool = False,
    zero_floor: bool = False,
    shared_scale: bool | Sequence[Sequence[int | str]] = False,
    colorbar_mode: str = "per-panel",
    colorbar_label: str | None = None,
    vmin: Any = None,
    vmax: Any = None,
    image_scale: str = "linear",
    image_interpolation: str = "hanning",
):
    if not images:
        raise ValueError("At least one detector image is required.")
    if len(titles) != len(images):
        raise ValueError(
            f"titles and images must have the same length: {len(titles)} != {len(images)}"
        )
    if colorbar_mode not in {"per-panel", "shared"}:
        raise ValueError("colorbar_mode must be 'per-panel' or 'shared'.")
    fig, axes = _detector_grid_axes(len(images))
    scale_groups = _readout_scale_groups(shared_scale, titles, len(images))
    if colorbar_mode == "shared" and len(scale_groups) != 1:
        raise ValueError(
            "colorbar_mode='shared' requires one shared scale group for all images."
        )
    scale_limits = _readout_group_limits(
        images,
        titles,
        scale_groups,
        clip=clip,
        symmetric=symmetric,
        zero_floor=zero_floor,
        vmin=vmin,
        vmax=vmax,
    )
    for idx, (ax, title, data) in enumerate(
        zip(axes.flat, titles, images, strict=False),
    ):
        vmin, vmax, clip_label, is_shared = scale_limits[idx]
        norm = _readout_image_norm(
            data,
            image_scale=image_scale,
            vmin=vmin,
            vmax=vmax,
        )
        im = ax.imshow(
            data,
            origin="lower",
            norm=norm,
            cmap=cmap,
            interpolation=image_interpolation,
            resample=True,
        )
        ax.set_title(f"{title}{title_suffix}", pad=8)
        if annotate_delta or annotate_stats:
            finite = data[np.isfinite(data)]
            max_abs = np.nanmax(np.abs(finite)) if finite.size else 0.0
            max_pos = np.nanmax(finite) if finite.size else 0.0
            min_delta = np.nanmin(finite) if finite.size else 0.0
            mean_value = np.nanmean(finite) if finite.size else 0.0
            median_value = np.nanmedian(finite) if finite.size else 0.0
            if annotate_delta:
                annotation = (
                    f"max {max_pos:.3g}\n"
                    f"min {min_delta:.3g}\n"
                    f"|max| {max_abs:.3g}"
                )
            else:
                annotation = (
                    f"median {median_value:.3g}\n"
                    f"mean {mean_value:.3g}\n"
                    f"max {max_pos:.3g}"
                )
            ax.text(
                0.02,
                0.96,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "black",
                    "ec": "none",
                    "alpha": 0.55,
                },
            )
        ax.axis("off")
        if colorbar_mode == "per-panel":
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            if colorbar_label:
                cbar.set_label(colorbar_label)
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    if colorbar_mode == "shared":
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    _center_detector_axes(axes, len(images))
    return fig, axes


def plot_detector_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    *,
    clip: float | None = 0.995,
    shared_scale: bool | Sequence[Sequence[int | str]] = False,
    colorbar_mode: str = "per-panel",
    colorbar_label: str | None = None,
    cmap: str = "viridis",
    title_suffix: str = "",
    symmetric: bool = False,
    zero_floor: bool = True,
    annotate_stats: bool = False,
    vmin: Any = None,
    vmax: Any = None,
    image_scale: str = "linear",
    image_interpolation: str = "hanning",
):
    """Plot detector-shaped image arrays with explicit scale/colorbar control."""
    image_arrays = [np.asarray(image, dtype=float) for image in images]
    titles = list(titles) if titles is not None else [
        f"detector {idx}" for idx in range(len(image_arrays))
    ]
    return _plot_detector_image_grid(
        image_arrays,
        titles,
        clip=clip,
        symmetric=symmetric,
        cmap=cmap,
        title_suffix=title_suffix,
        annotate_stats=annotate_stats,
        zero_floor=zero_floor,
        shared_scale=shared_scale,
        colorbar_mode=colorbar_mode,
        colorbar_label=colorbar_label,
        vmin=vmin,
        vmax=vmax,
        image_scale=image_scale,
        image_interpolation=image_interpolation,
    )


def plot_readout_overview(
    hdul: Any,
    titles: list[str] | None = None,
    *,
    clip: float | None = 0.995,
    shared_scale: bool | Sequence[Sequence[int | str]] = False,
    annotate_stats: bool = False,
    colorbar_mode: str = "per-panel",
    colorbar_label: str | None = None,
    cmap: str = "cividis",
    zero_floor: bool = True,
    vmin: Any = None,
    vmax: Any = None,
    image_scale: str = "linear",
    image_interpolation: str = "hanning",
):
    """Plot detector readout images from a ScopeSim readout result."""
    readouts = list(hdul)
    titles = titles or [f"detector {idx}" for idx in range(len(readouts))]
    images = [_readout_image_data(channel_hdul) for channel_hdul in readouts]
    return _plot_detector_image_grid(
        images,
        titles,
        clip=clip,
        symmetric=False,
        cmap=cmap,
        zero_floor=zero_floor,
        shared_scale=shared_scale,
        annotate_stats=annotate_stats,
        colorbar_mode=colorbar_mode,
        colorbar_label=colorbar_label,
        vmin=vmin,
        vmax=vmax,
        image_scale=image_scale,
        image_interpolation=image_interpolation,
    )


def plot_readout_delta_overview(
    signal_hdul: Any,
    reference_hdul: Any,
    titles: list[str] | None = None,
    *,
    clip: float | None = 0.995,
    title_suffix: str = " - reference",
    shared_scale: bool | Sequence[Sequence[int | str]] = False,
    colorbar_mode: str = "per-panel",
    colorbar_label: str | None = None,
    cmap: str = "magma",
    zero_floor: bool = True,
    vmin: Any = None,
    vmax: Any = None,
    image_scale: str = "linear",
    image_interpolation: str = "hanning",
    annotate_delta: bool = False,
):
    """Plot source-minus-reference detector readout images."""
    signal_readouts = list(signal_hdul)
    titles = titles or [f"detector {idx}" for idx in range(len(signal_readouts))]
    images = _readout_delta_images(signal_readouts, reference_hdul)
    return _plot_detector_image_grid(
        images,
        titles,
        clip=clip,
        symmetric=False,
        cmap=cmap,
        title_suffix=title_suffix,
        annotate_delta=annotate_delta,
        zero_floor=zero_floor,
        shared_scale=shared_scale,
        colorbar_mode=colorbar_mode,
        colorbar_label=colorbar_label,
        vmin=vmin,
        vmax=vmax,
        image_scale=image_scale,
        image_interpolation=image_interpolation,
    )


def plot_detector_cross_dispersion_cut(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    *,
    central_columns: int = 50,
    title_suffix: str = "",
    ylabel: str = "Median value",
):
    """Plot row profiles from the median of central detector columns."""
    import matplotlib.pyplot as plt

    image_arrays = [np.asarray(image, dtype=float) for image in images]
    titles = list(titles) if titles is not None else [
        f"detector {idx}" for idx in range(len(image_arrays))
    ]
    ncols = min(3, max(1, len(image_arrays)))
    nrows = int(np.ceil(len(image_arrays) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, title, data in zip(axes.flat, titles, image_arrays, strict=False):
        nx = data.shape[1]
        ncut = max(1, min(int(central_columns), 50, nx))
        x0 = nx // 2 - ncut // 2
        x1 = x0 + ncut
        profile = np.nanmedian(data[:, x0:x1], axis=1)
        ax.plot(np.arange(profile.size), profile, lw=1.8, color="tab:blue")
        ax.set_title(f"{title}{title_suffix} central {ncut} cols", pad=8)
        ax.set_xlabel("Detector row [pix]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(image_arrays):]:
        ax.axis("off")
    return fig, axes


def plot_readout_cross_dispersion_cut(
    hdul: Any,
    titles: list[str] | None = None,
    *,
    central_columns: int = 50,
    ylabel: str = "Median value",
):
    """Plot row profiles from the median of central detector columns."""
    readouts = list(hdul)
    titles = titles or [f"detector {idx}" for idx in range(len(readouts))]
    images = [_readout_image_data(channel_hdul) for channel_hdul in readouts]
    return plot_detector_cross_dispersion_cut(
        images,
        titles=titles,
        central_columns=central_columns,
        ylabel=ylabel,
    )


def _set_figure_suptitle(figure: Any, title: str | None) -> None:
    if title:
        figure.suptitle(title)


def show_and_save_hdul(
    hdul: Any,
    *,
    label: str,
    titles: Sequence[str] | None = None,
    reference_hdul: Any | None = None,
    hdul_view: Mapping[str, Any] | None = None,
    delta_view: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    show_hdul: bool | None = None,
    show_delta: bool | None = None,
    show_cross_dispersion: bool = False,
    save: bool = True,
    save_hdul: bool | None = None,
    save_reference: bool | None = None,
    save_delta: bool | None = None,
    save_figures: bool | None = None,
    figure_title: str | None = None,
    hdul_figure_title: str | None = None,
    delta_figure_title: str | None = None,
    cross_dispersion_figure_title: str | None = None,
    hdul_clip: float | None = None,
    hdul_shared_scale: bool | Sequence[Sequence[int | str]] = False,
    hdul_colorbar_mode: str = "per-panel",
    hdul_colorbar_label: str | None = None,
    hdul_cmap: str = "viridis",
    hdul_zero_floor: bool = False,
    hdul_vmin: Any = None,
    hdul_vmax: Any = None,
    hdul_image_scale: str = "linear",
    hdul_image_interpolation: str = "hanning",
    hdul_annotate: bool = False,
    delta_clip: float | None = 0.995,
    delta_shared_scale: bool | Sequence[Sequence[int | str]] = False,
    delta_colorbar_mode: str = "per-panel",
    delta_colorbar_label: str | None = None,
    delta_cmap: str = "magma",
    delta_zero_floor: bool = True,
    delta_vmin: Any = None,
    delta_vmax: Any = None,
    delta_image_scale: str = "linear",
    delta_image_interpolation: str = "hanning",
    delta_annotate: bool = False,
    delta_title_suffix: str = " - reference",
    cross_dispersion_central_columns: int = 50,
    cross_dispersion_ylabel: str = "Median value",
) -> dict[str, Any]:
    """Show and optionally save a detector readout, reference, and delta.

    ``save=True`` saves all products relevant to the call. ``save=False``
    disables all writes; the specific ``save_*`` flags only narrow
    ``save=True``.
    """
    hdul_clip = _view_value(hdul_view, "clip", default=hdul_clip)
    hdul_shared_scale = _view_value(
        hdul_view, "shared_scale", default=hdul_shared_scale,
    )
    hdul_colorbar_mode = _view_value(
        hdul_view, "colorbar_mode", default=hdul_colorbar_mode,
    )
    hdul_colorbar_label = _view_value(
        hdul_view, "colorbar", "colorbar_label", default=hdul_colorbar_label,
    )
    hdul_cmap = _view_value(hdul_view, "cmap", default=hdul_cmap)
    hdul_zero_floor = _view_value(
        hdul_view, "zero_floor", default=hdul_zero_floor,
    )
    hdul_vmin = _view_value(hdul_view, "vmin", default=hdul_vmin)
    hdul_vmax = _view_value(hdul_view, "vmax", default=hdul_vmax)
    hdul_image_scale = _view_value(
        hdul_view, "stretch", "image_scale", default=hdul_image_scale,
    )
    hdul_image_interpolation = _view_value(
        hdul_view, "interpolation", "image_interpolation",
        default=hdul_image_interpolation,
    )
    hdul_annotate = _view_value(hdul_view, "annotate", default=hdul_annotate)

    delta_clip = _view_value(delta_view, "clip", default=delta_clip)
    delta_shared_scale = _view_value(
        delta_view, "shared_scale", default=delta_shared_scale,
    )
    delta_colorbar_mode = _view_value(
        delta_view, "colorbar_mode", default=delta_colorbar_mode,
    )
    delta_colorbar_label = _view_value(
        delta_view, "colorbar", "colorbar_label", default=delta_colorbar_label,
    )
    delta_cmap = _view_value(delta_view, "cmap", default=delta_cmap)
    delta_zero_floor = _view_value(
        delta_view, "zero_floor", default=delta_zero_floor,
    )
    delta_vmin = _view_value(delta_view, "vmin", default=delta_vmin)
    delta_vmax = _view_value(delta_view, "vmax", default=delta_vmax)
    delta_image_scale = _view_value(
        delta_view, "stretch", "image_scale", default=delta_image_scale,
    )
    delta_image_interpolation = _view_value(
        delta_view, "interpolation", "image_interpolation",
        default=delta_image_interpolation,
    )
    delta_annotate = _view_value(
        delta_view, "annotate", default=delta_annotate,
    )

    readouts = list(hdul)
    reference_readouts = list(reference_hdul) if reference_hdul is not None else None
    titles = list(titles) if titles is not None else [
        f"detector {idx}" for idx in range(len(readouts))
    ]
    if len(titles) != len(readouts):
        raise ValueError(
            f"titles and hdul must have the same length: {len(titles)} != {len(readouts)}"
        )
    if reference_readouts is not None and len(reference_readouts) != len(readouts):
        raise ValueError(
            "reference_hdul and hdul contain different readout counts: "
            f"{len(reference_readouts)} != {len(readouts)}"
        )

    if show_hdul is None:
        show_hdul = reference_hdul is None
    if show_delta is None:
        show_delta = reference_hdul is not None

    if not save:
        save_hdul = False
        save_reference = False
        save_delta = False
        save_figures = False
    else:
        if save_hdul is None:
            save_hdul = True
        if save_reference is None:
            save_reference = reference_hdul is not None
        if save_delta is None:
            save_delta = reference_hdul is not None
        if save_figures is None:
            save_figures = True

    if reference_hdul is None and (show_delta or save_reference or save_delta):
        raise ValueError("reference_hdul is required to show or save reference/delta outputs.")

    files: dict[str, list[Path]] = {
        "hdul": [],
        "reference": [],
        "delta": [],
        "figures": [],
    }
    figures: dict[str, Any] = {
        "hdul": None,
        "delta": None,
        "cross_dispersion": None,
    }

    needs_output_dir = (
        save_hdul
        or save_reference
        or save_delta
        or (save_figures and (show_hdul or show_delta or show_cross_dispersion))
    )
    readout_dir = None
    if needs_output_dir:
        if output_dir is None:
            raise ValueError("output_dir is required when saving outputs.")
        readout_dir = Path(output_dir) / "readouts" / label
        readout_dir.mkdir(parents=True, exist_ok=True)

    delta_images = (
        _readout_delta_images(readouts, reference_readouts)
        if reference_readouts is not None else None
    )

    if show_hdul:
        fig_hdul, _axes = plot_readout_overview(
            readouts,
            titles=titles,
            clip=hdul_clip,
            shared_scale=hdul_shared_scale,
            annotate_stats=hdul_annotate,
            colorbar_mode=hdul_colorbar_mode,
            colorbar_label=hdul_colorbar_label,
            cmap=hdul_cmap,
            zero_floor=hdul_zero_floor,
            vmin=hdul_vmin,
            vmax=hdul_vmax,
            image_scale=hdul_image_scale,
            image_interpolation=hdul_image_interpolation,
        )
        _set_figure_suptitle(fig_hdul, hdul_figure_title or figure_title)
        figures["hdul"] = fig_hdul

    if show_delta:
        fig_delta, _axes = plot_readout_delta_overview(
            readouts,
            reference_readouts,
            titles=titles,
            clip=delta_clip,
            title_suffix=delta_title_suffix,
            shared_scale=delta_shared_scale,
            annotate_delta=delta_annotate,
            colorbar_mode=delta_colorbar_mode,
            colorbar_label=delta_colorbar_label,
            cmap=delta_cmap,
            zero_floor=delta_zero_floor,
            vmin=delta_vmin,
            vmax=delta_vmax,
            image_scale=delta_image_scale,
            image_interpolation=delta_image_interpolation,
        )
        _set_figure_suptitle(fig_delta, delta_figure_title or figure_title)
        figures["delta"] = fig_delta

    if show_cross_dispersion:
        if delta_images is None:
            cut_images = [_readout_image_data(channel_hdul) for channel_hdul in readouts]
            cut_suffix = ""
        else:
            cut_images = delta_images
            cut_suffix = delta_title_suffix
        fig_cut, _axes = plot_detector_cross_dispersion_cut(
            cut_images,
            titles=titles,
            central_columns=cross_dispersion_central_columns,
            title_suffix=cut_suffix,
            ylabel=cross_dispersion_ylabel,
        )
        _set_figure_suptitle(
            fig_cut,
            cross_dispersion_figure_title or figure_title,
        )
        figures["cross_dispersion"] = fig_cut

    if readout_dir is not None and save_figures:
        for figure_name, figure in figures.items():
            if figure is None:
                continue
            figure_path = readout_dir / f"{label}_{figure_name}.png"
            figure.savefig(figure_path, dpi=300)
            files["figures"].append(figure_path)

    if readout_dir is not None and save_hdul:
        for title, channel_hdul in zip(titles, readouts, strict=True):
            path = readout_dir / f"{label}_hdul_{title}.fits"
            _write_readout_product(path, channel_hdul)
            files["hdul"].append(path)

    if readout_dir is not None and save_reference:
        for title, channel_hdul in zip(titles, reference_readouts, strict=True):
            path = readout_dir / f"{label}_reference_{title}.fits"
            _write_readout_product(path, channel_hdul)
            files["reference"].append(path)

    if readout_dir is not None and save_delta:
        from astropy.io import fits

        for title, image, channel_hdul in zip(titles, delta_images, readouts, strict=True):
            header = getattr(_readout_image_hdu(channel_hdul), "header", None)
            image_hdu = fits.ImageHDU(
                data=np.asarray(image, dtype=float),
                header=header.copy() if header is not None else None,
                name="DELTA",
            )
            path = readout_dir / f"{label}_delta_{title}.fits"
            fits.HDUList([fits.PrimaryHDU(), image_hdu]).writeto(
                path,
                overwrite=True,
            )
            files["delta"].append(path)

    return {
        "figures": figures,
        "files": files,
        "directory": readout_dir,
    }


def plot_resolving_power_echellogram(
    table: Table,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    trace_width_fraction: float = 0.2,
):
    """Plot resolving power on the trace effect's physical focal plane."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    image_plane_ids = sorted({int(value) for value in table["image_plane_id"]})
    fig, axes = _detector_grid_axes(len(image_plane_ids), row_height=3.8)
    values = table["resolving_power_R"] / 1000.0
    vmin = np.min(values) if vmin is None else vmin
    vmax = np.max(values) if vmax is None else vmax
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else 0.1 * abs(vmin)
        vmin, vmax = vmin - pad, vmax + pad
    norm = Normalize(vmin=vmin, vmax=vmax)

    colormap = plt.get_cmap(cmap).copy()
    artist = None
    image_plane_array = table["image_plane_id"]
    plane_collections = []
    plane_centres = []

    for ax, image_plane_id in zip(axes.flat, image_plane_ids, strict=False):
        rows = table[image_plane_array == image_plane_id]
        row_values = values[image_plane_array == image_plane_id]
        x_mm = rows["detector_x_mm"]
        y_mm = rows["detector_y_mm"]

        trace_ids = rows["trace_id"]
        collections = []
        centres = []
        for trace_id in sorted({str(value) for value in trace_ids}):
            trace_mask = trace_ids == trace_id
            trace_rows = rows[trace_mask]
            trace_values = row_values[trace_mask]
            order = np.argsort(trace_rows["sample_index"])
            x = trace_rows["detector_x_mm"][order]
            y = trace_rows["detector_y_mm"][order]
            trace_values = trace_values[order]
            points = np.column_stack((x, y))
            midpoints = (points[:-1] + points[1:]) / 2
            starts = np.concatenate((points[:1], midpoints))
            ends = np.concatenate((midpoints, points[-1:]))
            segments = np.stack((starts, points, ends), axis=1)
            collection = LineCollection(
                segments,
                cmap=colormap,
                norm=norm,
                linewidths=1.1,
                zorder=2,
            )
            collection.set_array(trace_values)
            ax.add_collection(collection)
            artist = collection
            collections.append(collection)
            centres.append(points[len(points) // 2])

        plane_collections.append(collections)
        plane_centres.append(centres)

        channels = sorted({str(value) for value in rows["channel"]})
        resolving_power = rows["resolving_power_R"]
        seeing = np.median(rows["seeing_fwhm_arcsec"])
        spatial = np.median(rows["spatial_fwhm_pix"])
        diagnostic = (
            f"R {np.min(resolving_power) / 1000.0:.1f}-"
            f"{np.max(resolving_power) / 1000.0:.1f}k"
        )
        ax.set_title(
            f"{'/'.join(channels)}\n"
            f"{diagnostic}, {seeing:.2f}\"/{spatial:.1f} pix "
            "$FWHM_{spat}$",
            pad=8,
        )
        x_pad = max(0.05 * np.ptp(x_mm), 0.05)
        y_pad = max(0.05 * np.ptp(y_mm), 0.05)
        ax.set_xlim(np.min(x_mm) - x_pad, np.max(x_mm) + x_pad)
        ax.set_ylim(np.min(y_mm) - y_pad, np.max(y_mm) + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, lw=0.5)
        ax.set_xlabel("Focal-plane x [mm]")
        ax.set_ylabel("Focal-plane y [mm]")

    for ax in axes.flat[len(image_plane_ids):]:
        ax.axis("off")
    cbar = fig.colorbar(
        artist,
        ax=axes.ravel().tolist()[:len(image_plane_ids)],
        shrink=0.86,
        pad=0.02,
    )
    cbar.set_label("Resolving power [k]")

    _center_detector_axes(axes, len(image_plane_ids))
    fig.canvas.draw()
    for ax, collections, centres in zip(
        axes.flat, plane_collections, plane_centres, strict=False,
    ):
        if len(centres) < 2:
            continue
        centres = ax.transData.transform(centres)
        distances = np.linalg.norm(
            centres[:, None, :] - centres[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        spacing = np.median(np.min(distances, axis=1))
        linewidth = trace_width_fraction * spacing * 72 / fig.dpi
        for collection in collections:
            collection.set_linewidth(linewidth)
    return fig, axes
