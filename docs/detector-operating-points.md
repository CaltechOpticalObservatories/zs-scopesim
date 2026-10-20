# ZShooter detector operating points: current assumptions and next specifications

The existing detector plots compare effective read-noise/dark/QE assumptions at a common cadence. They do not yet establish a feasible operating cadence for LmAPDs versus H2RGs. The independent-integration variance scaling is present; the missing constraint is usable well at the selected detector operating point.

## What the current calculations actually use

| Detector case | Effective read noise per integration [input electrons RMS] | Dark rate [electrons/hour/pixel] | One-hour forecast |
|---|---:|---:|---|
| Current qCCD | .1 | .5976 | 1800 s x 2 |
| Conventional CCD alternative | 2.5 | 2.0 | 1800 s x 2 |
| Current LmAPD | .5 | .36 | 1800 s x 2 |
| H2RG alternative | 6 | 23 | 1800 s x 2 |

These are configured engineering assumptions, not four measured vendor operating points. The performance notebook uses `NDIT=2`; the comparison notebook uses `NDIT=1` per readout and combines two independent integrations analytically. Both count two read-noise contributions. Samples within a ramp are not counted as independent integrations. The supplied effective RMS is intended to represent the completed CDS/ramp measurement.

For an independent equal-time sky subtraction, the current accumulated-electron formula is

\[
V=S+2(B+D+N\sigma_r^2).
\]

The native detector path applies Poisson noise to integrated source, sky and dark, then adds independent read noise. The comparison's [stochastic residual checks](../validation/xshooter/data/comparison/noisy_extraction.csv) support this implemented variance model. They do not test whether the selected integration length is feasible for the intended hardware.

The [saved peak counts](../validation/xshooter/data/comparison/saturation.csv) include **572,093 electrons per 1800 s H integration** in the natural/.70 case, and 308,522 in K. ZShooter's 64,000-electron linearity limit is a disabled placeholder; there is no gain-dependent usable-well or reset model. Those frames cannot currently justify an operational 1800 s NIR cadence. Changing the number of independent integrations increases read variance by \(N\sigma_r^2\), even when the total photon exposure is unchanged.

## Why the APD QE plot is easy to misread

`irdb/ZShooter_v2/detector_specs/APD_TER.dat` explicitly describes an LmAPD curve based on H4RG data with a “1.2x excess noise factor.” Its values are the alternative `H2RG_TER.dat` curve divided by 1.2 at all 2,469 matching wavelengths. The latter file labels itself an estimate and placeholder inherited from METIS. These are not two independently specified vendor QE curves.

Consequently the displayed conventional/current NIR QE ratio of 1.2 is the deliberately imposed excess-noise proxy. In the photon-background-dominated limit that proxy produces a .099 mag disadvantage, while read-noise improvements have little effect. This helps explain modest H gains or K losses without implying that the hardware trade has been decided.

Folding excess noise into QE also changes the simulated mean counts. It is only equivalent to an explicit variance factor in the photon-noise-dominated limit. If \(F\) denotes the **variance multiplier**, with physical counts \(S,B\), then reduced QE with unchanged input-referred read variance gives

\[
{\rm S/N}_{\rm proxy}=\frac{S}{\sqrt{F(S+2B)+F^2V_r}},
\]

where the corresponding physical-count model gives \(S/\sqrt{F(S+2B)+V_r}\). Dark terms also require consistent treatment. Before separating this approximation, establish whether the original quoted factor 1.2 referred to variance or RMS; they differ by a square. No physical assumptions have been changed in this assessment.

## Minimum operating-point information for the vendor discussion

The practical specification is a small table of selectable operating points, not a detailed ramp or device simulator:

| Input | Convention needed |
|---|---|
| Physical QE versus wavelength | Detection probability before multiplication; keep separate from excess noise. |
| Avalanche gain and excess-noise factor | State \(M\) and whether \(F\) multiplies variance; give dependence on bias/temperature or provide selected design points. |
| Effective read noise | Input-referred electrons RMS for a completed CDS/ramp, with integration time, sample pattern and readout speed. Do not divide an already input-referred value by gain again. |
| Dark signal | Input-referred mean and variance at the operating point; distinguish multiplied and unmultiplied contributions if the vendor reports them separately. |
| Usable well/linearity limit | Input-referred electrons per integration at that gain, with the acceptable nonlinearity; distinguish detector charge capacity from output-node/ADC limits. |
| Electronic conversion | Output electrons per ADU, bias convention and ADC range. This gain is distinct from avalanche multiplication. |
| Timing | Minimum useful integration, reset/read overhead and duty cycle. For qCCDs, specify the sampling/readout mode associated with low RN and its time cost. |
| Binning | Allowed CCD charge binning and its well/readout limits; keep separate from software summation of independently read pixels. |

Effective input-referred RN, dark and usable well are enough for a useful first model. Detailed correlated ramp noise and Pyxel are unnecessary for this step. Missing hardware values should remain explicit design-point axes or parameter sweeps until specified, rather than being replaced by an asserted “typical” number.

As a units check, a simplified APD with pre-multiplication detected charge \(S+B\), avalanche gain \(M\), and output conversion \(g\) electrons/ADU has mean signal \(M(S+B)/g\) ADU. Its photon variance is \(M^2F(S+B)/g^2\), not Poisson with that ADU mean. Effective input-referred quantities are a convenient way to retain these distinctions without simulating individual avalanches. Dark charge generated before and after multiplication need not transform identically.

## The next fair comparison

Reuse the simulated optical rates. For each detector operating point, calculate the allowed DIT from the peak physical source+sky+thermal charge and its usable-well limit, then count independent integrations needed for the requested on-source time. Apply the chosen headroom and overhead explicitly. APDs and H2RGs should have their own feasible cadences; equal NDIT is appropriate only when imposed by an observing constraint.

Show the existing five-minute/hour and native/R1000 cases first with common physical QE to isolate noise/cadence, then with the selected QE assumptions. Include sky, source, dark and read contributions. Report both on-source and wall-clock exposure where resets/readouts differ. Repeat for qCCD/CCD with charge binning and readout-speed constraints.

Verification should cover mean/variance in electrons and ADU, gain conversion, summed independent integrations versus averaged products, saturation **per integration**, and limiting cases where read noise or background dominates. The helper-header problem is corrected: `save_zshooter_readout` now obtains RN/dark and accumulated exposure from the configured readout rather than hardcoded values. The kilonova reduction inputs carry accumulated electrons and a separate expected pixel-variance image, with native ADU/averaged outputs preserved separately. Focused tests check gain conversion, summed versus averaged products, zero-DIT bias frames, metadata and shared-sky coefficients.

The earlier background-budget “fraction of full well” column also used accumulated DIT x NDIT additive background against a one-integration well and omitted source/OH peaks from that particular budget. Use the full rate-image peak per integration for this engineering decision, not that column.
