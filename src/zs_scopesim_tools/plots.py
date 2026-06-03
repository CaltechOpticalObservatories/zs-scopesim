"""Plotting helpers for ZShooter ScopeSim validation notebooks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM


def _as_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


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
                _plot_quantity_values(spectrum),
                lw=1.4,
                color=colors.get(name, "0.5"),
                label=f"{name} diffuse",
            )
        if channel.get("total_spectrum_without_blocking") is not None:
            ax.plot(
                wave,
                _plot_quantity_values(
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
                _plot_quantity_values(channel["total_spectrum"]),
                lw=2.2,
                color="black",
                alpha=0.95,
                label="total with ir block",
            )
        ax.set_title(
            f"{channel['label']} (image plane {channel['image_plane_id']}): "
            f"{channel['total_rate_ph_s_pix']:.3g} ph/s/pix"
            f" ({channel.get('blocking_delta_rate_ph_s_pix', 0.0):.3g} blocked)",
        )
        ax.set_xlim(wave.min(), wave.max())
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spectral background [PHOTLAM equiv.]")
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
               channel["post_disperser_after_qe"],
               channel.get("post_disperser_without_blocking_after_qe",
                           np.zeros_like(channel["post_disperser_after_qe"]))]
        )
        for curve in curves:
            finite = curve[np.isfinite(curve)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(finite)))
    ymax = max(1e-3, ymax * 1.08)

    for ax, (aperture_id, channel) in zip(axes.flat, data["channels"].items()):
        for name, values in channel["pre_disperser_terms"].items():
            ax.plot(
                wave, values, lw=1.0, ls="--", alpha=0.75,
                color=group_colors.get(name, "0.5"),
                label=f"pre {name}",
            )
        for name, values in channel["post_disperser_terms"].items():
            ax.plot(
                wave, values, lw=1.1, ls="-", alpha=0.7,
                color=group_colors.get(name, "0.5"),
                label=f"post {name}",
            )

        ax.plot(
            wave, channel["pre_disperser_output_equiv"], lw=2.0,
            color="tab:purple", alpha=0.9,
            label="pre total before trace QE", zorder=8,
        )
        ax.plot(
            wave,
            channel.get(
                "post_disperser_without_blocking_after_qe",
                channel["post_disperser_after_qe"],
            ),
            lw=2.0,
            ls="--",
            color="0.25",
            alpha=0.85,
            label="post diffuse without ir block",
            zorder=9,
        )
        ax.plot(
            wave, channel["post_disperser_after_qe"], lw=2.4,
            color="black", alpha=0.95,
            label="post diffuse with ir block+QE", zorder=10,
        )
        blocked_delta = channel.get("post_disperser_blocked_delta")
        if blocked_delta is not None and np.nanmax(np.abs(blocked_delta)) > 0:
            ax.plot(
                wave, blocked_delta, lw=1.8, ls="-.",
                color="tab:brown", alpha=0.9,
                label="removed by ir block", zorder=9,
            )
        extract_curve = channel.get("post_disperser_extract_equiv_rate_ph_s")
        if extract_curve is not None:
            rate_ax = ax.twinx()
            rate_ax.plot(
                wave,
                extract_curve,
                lw=1.2,
                ls=":",
                color="tab:orange",
                alpha=0.8,
                label="integrated diffuse extraction equiv.",
            )
            ax.plot(
                [], [], lw=1.2, ls=":", color="tab:orange",
                alpha=0.8, label="integrated diffuse extraction equiv.",
            )
            rate_ax.set_yscale("symlog", linthresh=1e-6)
            rate_ax.tick_params(axis="y", labelsize=7, colors="tab:orange")
            if aperture_id in (2, 5):
                rate_ax.set_ylabel("Integrated diffuse [ph/s]", color="tab:orange")
            peak_rate = float(np.nanmax(extract_curve))
        else:
            peak_rate = np.nan
        peak_lines = []
        for label, values in (
            ("pre", channel["pre_disperser_output_equiv"]),
            ("post", channel["post_disperser_after_qe"]),
            ("no block", channel.get(
                "post_disperser_without_blocking_after_qe",
                channel["post_disperser_after_qe"],
            )),
        ):
            finite = values[np.isfinite(values)]
            if finite.size:
                peak_lines.append(f"{label}: {np.nanmax(finite):.2g}")
        if np.isfinite(peak_rate):
            peak_lines.append(f"int: {peak_rate:.2g} ph/s")
        ax.text(
            0.02,
            0.04,
            "peak " + "\n".join(peak_lines),
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                  "ec": "0.75", "alpha": 0.78},
        )
        ax.plot(
            wave, channel["detector_qe"], lw=0.9, ls=":",
            color="tab:red", alpha=0.8, label="QE throughput",
        )
        ax.set_title(f"{channel['label']} (aperture {aperture_id})")
        ax.set_xlim(wave.min(), wave.max())
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Thermal emission density [PHOTLAM equiv.] (symlog)")

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
            wave, channel["detector_qe"], lw=1.8, ls=":",
            color="tab:red", alpha=0.9,
            label=channel.get("detector_qe_label", "detector QE"),
        )
        midpoint_qe = channel.get("detector_qe_midpoint")
        if midpoint_qe is not None:
            ax.plot(
                wave, midpoint_qe, lw=1.4, ls="--",
                color="tab:red", alpha=0.75,
                label=channel.get(
                    "detector_qe_midpoint_label",
                    "detector QE midpoint",
                ),
            )

        for idx, (_trace_id, order) in enumerate(channel["orders"].items()):
            order_label = "disperser/order" if idx == 0 else None
            order_qe_label = "QE at order trace" if idx == 0 else None
            total_label = "total/order" if idx == 0 else None
            ax.plot(
                wave, order["disperser"], lw=0.7, color="tab:orange",
                alpha=0.35, label=order_label,
            )
            ax.plot(
                wave, order["detector_qe"], lw=1.0, color="tab:pink",
                alpha=0.35, label=order_qe_label,
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
        (-0.5 * width, -0.5 * length),
        width,
        length,
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
    pad = max(width, np.ptp(x) if x.size > 1 else width, 0.25) * 0.7
    ax.set_xlim(min(-width, x.min()) - pad, max(width, x.max()) + pad)
    ax.set_ylim(min(-0.5 * length, y.min()) - pad,
                max(0.5 * length, y.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Across slit [arcsec]")
    ax.set_ylabel("Along slit [arcsec]")
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
                (-0.5 * width, -0.5 * length),
                width,
                length,
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
                ax.set_xlabel("Across slit [arcsec]")
            if col == 0:
                ax.set_ylabel("Along slit [arcsec]")
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


def plot_slit_loss_by_arm(data: Mapping[str, Any]):
    """Plot centered point-source slit loss for each spectrograph arm."""
    import matplotlib.pyplot as plt

    arms = data["arms"]
    fig, axes = plt.subplots(
        1,
        len(arms),
        figsize=(6.3 * len(arms), 4.4),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "zenith": "tab:blue",
        "elevation_60_ad_only": "tab:orange",
        "elevation_60_adc_residual": "tab:green",
    }
    for ax, (arm_name, arm) in zip(axes.flat, arms.items(), strict=True):
        wave = u.Quantity(arm["wave_nm"]).to_value(u.nm)
        for curve_name, curve in arm["curves"].items():
            ax.plot(
                wave,
                curve["loss"],
                lw=2.8,
                color=colors.get(curve_name, None),
                label=curve["label"],
            )
        ax.set_title(
            f"{arm_name}: {arm['slit_width_arcsec'].to_value(u.arcsec):.2f} arcsec slit"
        )
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Slit loss fraction")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=len(handles),
        frameon=False,
    )
    fig.suptitle(
        "Centered Point-Source Slit Loss "
        f"(seeing {data['seeing_arcsec'].to_value(u.arcsec):.2f} arcsec)",
        fontsize=12,
    )
    return fig, axes


def plot_readout_overview(hdul: Any, titles: list[str] | None = None):
    """Plot detector readout images from a ScopeSim readout result."""
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    readouts = list(hdul)
    titles = titles or [f"detector {idx}" for idx in range(len(readouts))]
    ncols = min(3, max(1, len(readouts)))
    nrows = int(np.ceil(len(readouts) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    interval = ZScaleInterval()
    for ax, title, channel_hdul in zip(axes.flat, titles, readouts, strict=False):
        image_hdu = (
            channel_hdul
            if hasattr(channel_hdul, "data")
            else channel_hdul[1]
        )
        data = np.asarray(image_hdu.data, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size:
            vmin, vmax = interval.get_limits(data)
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(
            data,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="cividis",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    for ax in axes.flat[len(readouts):]:
        ax.axis("off")
    return fig, axes
