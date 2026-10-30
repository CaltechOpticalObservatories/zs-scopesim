# Caltech simulator review series

The 29 pull requests preserve the approved thematic order across five CaltechOpticalObservatories forks. Their first entries target `main`, `dev_master` or `master`; later entries target the preceding review branch in the same repository. The original X-shooter and ZImager working branches are retained as historical snapshots. No upstream PR was created.

New development continues from these published review tips. The [H4RG continuation](h4rg-development.md) lists the follow-up branches and the original broad PRs they supersede. The 29-part numbering and existing review branches remain unchanged.

Merge in the numbered order below, using merge commits so the pinned commits and descendant ancestry remain valid. After merging an entry, retarget the next entry in that repository to its default branch and check that its diff contains only its theme. Avoid squash/rebase merges while descendants depend on those commits. Nothing in this series is automatically merged.

Yashvi-Sharma is requested on every PR; cfremling is additionally requested on series 05, 07, 08, 09, 16 and 29. Series 29 is the workspace ZImager notebook/integration PR.

## Review order

| Series | Repository | Theme / PR | Base |
|---:|---|---|---|
| 01 | zs-scopesim | [Pin the deployed simulator baseline for the review series](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/11) | `main` |
| 02 | scopesim | [Integrate the adopted upstream ScopeSim baseline](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/24) | `main` |
| 03 | irdb | [Integrate the adopted upstream instrument baseline](https://github.com/CaltechOpticalObservatories/irdb/pull/12) | `dev_master` |
| 04 | PyReduce | [Update the Caltech reduction fork to the adopted PyReduce 0.9 baseline](https://github.com/CaltechOpticalObservatories/PyReduce/pull/1) | `master` |
| 05 | scopesim | [Preserve arm exposures, charge binning and independent detector readouts](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/25) | `review/xs-02-upstream-baseline` |
| 06 | scopesim | [Correct analytical echelle angles, placement and physical slit dispersion](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/26) | `review/xs-05-detector-readouts` |
| 07 | scopesim | [Use calibrated traces and physical slit and detector coordinates](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/27) | `review/xs-06-echelle-geometry` |
| 08 | scopesim | [Preserve native spectral photon integrals and source representations](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/28) | `review/xs-07-calibrated-traces` |
| 09 | scopesim | [Integrate slit illumination and detector pixels with pre-slit ADC shifts](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/29) | `review/xs-08-spectral-integrals` |
| 10 | scopesim | [Model configured avalanche noise and warn at detector exposure limits](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/30) | `review/xs-09-slit-projection` |
| 11 | irdb | [Add attributed X-shooter calibration inputs and preparation calculations](https://github.com/CaltechOpticalObservatories/irdb/pull/13) | `review/xs-03-upstream-baseline` |
| 12 | irdb | [Configure full-frame X-shooter spectroscopy and detector readouts](https://github.com/CaltechOpticalObservatories/irdb/pull/14) | `review/xs-11-xshooter-calibrations` |
| 13 | irdb | [Adopt the selected ZShooter optical prescription and physical slit settings](https://github.com/CaltechOpticalObservatories/irdb/pull/15) | `review/xs-12-xshooter-instrument` |
| 14 | irdb | [Configure editable ZShooter detector alternatives and operating points](https://github.com/CaltechOpticalObservatories/irdb/pull/16) | `review/xs-13-zshooter-prescription` |
| 15 | PyReduce | [Preserve joint slit-curvature fits and register wavelength search radius](https://github.com/CaltechOpticalObservatories/PyReduce/pull/2) | `review/xs-04-version-09` |
| 16 | PyReduce | [Propagate extraction covariance and converge faint slit profiles](https://github.com/CaltechOpticalObservatories/PyReduce/pull/3) | `review/xs-15-curvature-configuration` |
| 17 | zshooter-drp | [Repair ZShooter order identities, calibration ranges and extraction settings](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/5) | `main` |
| 18 | zshooter-drp | [Propagate prepared detector variance and physical spectral covariance](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/6) | `review/xs-17-zshooter-format` |
| 19 | zshooter-drp | [Reduce native X-shooter arms with their configured count conventions](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/7) | `review/xs-18-extraction-variance` |
| 20 | zshooter-drp | [Bind calibrations to detector geometry and store compressed FITS products](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/8) | `review/xs-19-xshooter-reduction` |
| 21 | zs-scopesim | [Use configured detector rates and covariance-aware readout helpers](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/12) | `review/xs-01-baseline-pins` |
| 22 | zs-scopesim | [Reconcile performance forecasts and compare configured detector packages](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/13) | `review/xs-21-scientific-helpers` |
| 23 | zs-scopesim | [Document accepted geometry and execute resolution and slit-illumination checks](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/14) | `review/xs-22-performance-trades` |
| 24 | zs-scopesim | [Compare flat-AB observations at native and common resolving power](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/15) | `review/xs-23-resolution-lamps` |
| 25 | zs-scopesim | [Mirror kilonova reductions and compare the simulated and published spectra](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/16) | `review/xs-24-flat-ab-comparison` |
| 26 | zs-scopesim | [Consolidate simulator validation and publish reproducible dependency pins](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/17) | `review/xs-25-kilonova-comparison` |
| 27 | scopesim | [Apply imaging branch throughput before wavelength integration](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/31) | `review/xs-10-detector-operating-points` |
| 28 | irdb | [Configure ZImager optics, filters and stochastic detector readouts](https://github.com/CaltechOpticalObservatories/irdb/pull/17) | `review/xs-14-alternate-detectors` |
| 29 | zs-scopesim | [Execute ZImager and verify unchanged spectroscopic simulations](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/18) | `review/xs-26-validation-reproducibility` |

## Verification of the split

The final ScopeSim, PyReduce and DRP production trees match the reviewed source snapshots exactly. IRDB production files also match; its old physical-test fixtures now resolve detector settings and trace inputs through the configured commands. The workspace differs only in published dependency pins, portable documentation links/publication wording, and these review aids. Notebook blobs, calibration inputs and saved scientific figures were retained unchanged. Long science notebooks were not rerun solely to reorganize their commits.

Tests run in isolated checkouts with the existing `zssim` interpreter and explicit import paths:

| Boundary | Result |
|---|---|
| ScopeSim upstream baseline | 100 passed |
| ScopeSim readouts | 93 passed |
| ScopeSim geometry/calibrated traces | 135 passed |
| ScopeSim spectral integrals | 45 passed, 2 expected failures |
| ScopeSim shared numerics/operating points | 249 passed, 3 expected failures |
| PyReduce focused curvature/configuration/covariance/trace/spectra | 89 passed; local C extension built |
| DRP intermediate variance boundary / final suite | 11 / 39 passed |
| X-shooter configured instrument | 36 passed |
| ZShooter configured physics | 22 repaired fixture cases passed; 7 unchanged scene cases passed |
| Workspace helper boundary / complete spectroscopy suite | 95 / 188 passed |
| Integrated ScopeSim imaging/Moffat/ADC | 74 passed |
| Integrated workspace imaging/detector/noise | 11 passed |

These suites overlap; their counts must not be summed as unique tests. Twelve additional PyReduce curvature integration cases could not acquire their external example archives and are not counted as passing. The first IRDB run identified stale detector/trace fixtures (7 failures and 12 setup errors); the detector-configuration PR repairs those fixtures without changing production physics. The sparse dark-only check now uses 90,000 samples per region while retaining its tolerances.

The preserved ZImager integration evidence compares 18 full spectroscopy rate arrays bit for bit. This remains the original source-branch execution; exact production-tree comparisons establish that the review stack contains the same implementation. The X-shooter faint-mode NIR absolute flux-calibration limitation remains flagged beside its science evidence.

## GitHub CI at publication

The local results above are separate from GitHub Actions. The new PRs are open for review, but CI is not uniformly passing. The inspected jobs identify three follow-up areas:

- [ScopeSim baseline offline job](https://github.com/CaltechOpticalObservatories/ScopeSim/actions/runs/40016217269/job/120628819529): test collection fails with `ModuleNotFoundError: No module named 'palace'`. The hosted runner needs the fork's PALACE dependency configured; the local `zssim` environment already has it.
- [Workspace documentation job](https://github.com/CaltechOpticalObservatories/zs-scopesim/actions/runs/40016572334/job/120630043859): `sphinx-build -W` fails on unresolved report/evidence links, including files outside the documentation source tree and references supplied by later PRs in the stack. Resolve those links for the built documentation and account for intermediate PR boundaries.
- [IRDB MICADO notebook job](https://github.com/CaltechOpticalObservatories/irdb/actions/runs/40016224858/job/120628895328): SkyCalc times out connecting to the ESO service. A fixed local sky input or an explicit policy for network-dependent jobs is needed for reliable CI.

These are representative job inspections, not a complete CI failure diagnosis. Other matrix jobs were still running at handoff. No workflow behavior, test selection or failure policy was changed during the PR split.

## Notebook review

The following files contain cell sources only, with output blobs and execution metadata excluded. They are review material; the corresponding committed notebooks retain their executed tables and figures.

- [Series 21: notebook source diff](review-series/21-notebook-sources.diff)
- [Series 22: notebook source diff](review-series/22-notebook-sources.diff)
- [Series 23: notebook source diff](review-series/23-notebook-sources.diff)
- [Series 24: notebook source diff](review-series/24-notebook-sources.diff)
- [Series 25: notebook source diff](review-series/25-notebook-sources.diff)
- [Series 26: notebook source diff](review-series/26-notebook-sources.diff)
- [Series 29: notebook source diff](review-series/29-notebook-sources.diff)
