# Continuing development after the review split

The published review series is the development history from this point forward.
New work branches from its last relevant PR, so development can continue before
that PR merges. The original 29-part series remains unchanged.

## Active branches and dependencies

| Repository | Development branch | Predecessor | Follow-up |
|---|---|---|---|
| PyReduce | `fix/altdet-extraction` | [#3](https://github.com/CaltechOpticalObservatories/PyReduce/pull/3) | [#4](https://github.com/CaltechOpticalObservatories/PyReduce/pull/4) |
| zshooter-drp | `fix/altdet-calibration` | [#8](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/8) | [#9](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/9) |
| IRDB | `feat/h4rg-k` | [#17](https://github.com/CaltechOpticalObservatories/irdb/pull/17) | [#18](https://github.com/CaltechOpticalObservatories/irdb/pull/18) |
| zs-scopesim | `feat/h4rg-comparison` | [#18](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/18) | This branch |
| ScopeSim | `review/zi-27-zimager-throughput` | [#31](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/31) | Unchanged dependency |

The DRP follow-up requires the PyReduce follow-up. Correctly labeled H4RG frames
require the new DRP mode settings. The workspace follow-up uses all three new
dependency commits, pinned in `env/zshooter-stack.toml`; the ScopeSim pin stays
at the existing review tip. PyReduce is published to the Caltech fork, with its
original `ivh/PyReduce` remote retained separately in the existing checkout.

Keep meaningful commits within these PRs: the workspace separates per-arm
helpers, the flat-AB comparison, the kilonova notebook and dependency/review
documentation. A new commit does not require another PR.

## Daily work and merging

Continue committing on the topic branches above. For a dependent next feature,
create a new branch from the relevant current topic branch and target that
predecessor in the new PR. Put requested review corrections on the affected PR
branch and merge them forward into dependent branches, then recheck tests and
dependency pins.

Merge predecessors before successors, using merge commits as required by the
original series. After a predecessor merges, retarget its immediate successor
to the repository's default branch and verify that only its intended changes
remain in the diff. Retain exact commit pins so another machine can use the
tested stack before its PRs merge.

Use the active branches for development. `scripts/sync_stack.sh` checks out the
exact manifest SHAs for reproduction and can leave a detached HEAD. Restart
notebook kernels after changing installed source checkouts.

## Transition verification

Before transferring the new work, the final review tips were compared with the
original development snapshots. ScopeSim, PyReduce and DRP trees matched
exactly. IRDB differed only in the review series' detector-test fixture repairs;
the workspace differed only in published dependency pins, documentation and
source-only review aids. Those review changes were retained.

The original combined PRs are superseded by this single review path:
[workspace #10](https://github.com/CaltechOpticalObservatories/zs-scopesim/pull/10),
[IRDB #11](https://github.com/CaltechOpticalObservatories/irdb/pull/11),
[ScopeSim #23](https://github.com/CaltechOpticalObservatories/ScopeSim/pull/23),
and [DRP #4](https://github.com/CaltechOpticalObservatories/zshooter-drp/pull/4).
Their original branches are retained. New local changes were first committed on
`preserve/pre-review-20261029` branches in the four changed repositories before
being transferred to the review history. Unrelated PRs and local data remain
outside this transition.

Validation on 2026-10-29: 175 focused tests passed against the transferred
Python code and configured detector packages: 86 PyReduce, 50 DRP, 36 workspace
helper/comparison tests and three IRDB detector-mode cases. Twelve PyReduce
curvature tests requiring external example datasets were excluded. Reported
warnings were synthetic FITS WCS warnings and the existing deprecated
`fov_grid` calls. These are local results, not a claim that all hosted CI passes.

The new IRDB cases verify the SPEC, H2RG and H4RG detector types, physical pixel
pitches, image dimensions, mode headers and eight modeled K orders. The DRP
checks exercise the H4RG settings lookup, retained mode identity and rejection
of incompatible saved calibrations.

## Preserved scientific results

The notebook files and comparison results were transferred without changing
their scientific sources or executed outputs. The [source-only notebook
diff](review-series/h4rg-notebook-sources.diff) compares the flat-AB notebook
against the review base and the new kilonova notebook against its original.
The original `01_Kilonovae.ipynb` remains unchanged.

The preceding kilonova execution retained 124 orders in each detector mode.
Its four-arm proof took 4m44s; the full ALTDET comparison took 21m23s and the
ALTDET4K comparison took 5m54s by reusing five identical arms. These are the
preserved source-run measurements, not new timings from this history transfer.
Two native iteration-limit warnings remained, with fixed-point refinement and
covariance checks completing successfully. An earlier gzip read failure was
not reproduced in five readbacks; its cause is not claimed fixed.

The H4RG metadata correction identifies future frames as `SPEC_ALTDETECT4K`
without changing physical geometry or noise parameters. Historical local raw
frames retain their original headers. The notebook's conservative model-file
freshness check can require fresh simulations after this configuration edit;
existing plotted spectra remain the saved execution evidence. No historical
raw-frame headers or timestamps were rewritten during the transition.
