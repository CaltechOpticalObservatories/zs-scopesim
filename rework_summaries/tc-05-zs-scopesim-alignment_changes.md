# tc-05-zs-scopesim-alignment

Follows ScopeSim branch `rework/tc-02-echelle-schema` and replaces drifted
copies of ScopeSim helpers. **The `zshooter_spie2026` and
`zshooter_performance` numbers do not move**: verified byte-identical on a
216-point fingerprint.

## 1. Detector placement uses the new API

`tests/test_xdisp_geometry.py` reached into a `SpectrographSetup` and set four
private attributes to place a model on a detector. ScopeSim now exposes
`set_slit_rotation()` and `set_detector_center()`; the test uses them.

`test_footprint_requires_the_trace_generators_placement` still asserts
`AttributeError`, now matching on the method name the error tells you to call.
A second test covers the other half: the rotation and the centre are configured
separately, so a model with a rotation but no centre must also fail clearly.

**`validation/zshooter_xdisp/geometry.py` is deliberately untouched.** It loads
`echelle.py` from pinned git revisions rather than the working tree, so the
objects it builds come from historical code that has no such property. Migrating
it to the new API breaks it — which is how this constraint was discovered.

## 2. Drifted QE copies replaced by delegation

`validation.py` carried private copies of ScopeSim helpers that had drifted:

| helper | drift |
|---|---|
| `representative_positional_qe` | dropped the `footprint` parameter that `illumination.py` forwards to a positional QE callable |
| `effective_diffuse_qe` | consequently differed in its second term |
| `_evaluate_diffuse_throughput` | stood in for `diffuse_detector_qe`, falling back to `evaluate_throughput` rather than `detector_qe_at`, so a position-aware QE without `effective_diffuse_throughput` lost its positional branch |

All three now delegate to ScopeSim, so this package cannot drift from the
simulator it exists to validate. `representative_positional_qe` gains the
`footprint` parameter it was missing; `_evaluate_diffuse_throughput` is deleted
as orphaned.

**This restores footprint forwarding, and the numbers still do not move** —
because the ZShooter configuration passes no `positional_qe`, so the dropped
branch was never reached. The drift was latent rather than active. Worth saying
plainly: it was a real divergence that happened not to bite yet.

## 3. Redundant `from_currsys` imports removed

`validation.py` imported `from scopesim.utils import from_currsys` five times —
once at module level and again as the first statement of `_resolved_for_display`,
`_required_cmd_value`, `_resolved_meta_summary` and `_resolved_detector_meta`.
The four local imports shadowed the module-level one with the same object, so
they were pure noise; they are deleted and all four functions now resolve the
name from module scope. There is no circular-import reason for them —
`validation.py` already imports from `scopesim.effects.illumination` at module
level.

Three function-local `from scopesim.effects.illumination import ...` statements
went the same way (`integrate_spectral_background` imported twice,
`image_plane_pixel_area` once), folded into the module-level import from that
same module at the top of the file.

In `helpers.py`, `SelectorWheel` was imported locally in two functions and
`detector_noise` in two more; both are now module-level.

## Some local imports are load-bearing

Hoisting every local import breaks nine tests, and the reason is worth
recording because the next person to tidy these will hit it again.

`tests/test_helpers.py` replaces `scopesim.utils.get_observation_info_from_cmds`
and `tests/test_reduction_readouts.py` replaces `comparison.detector_rates`.
Both substitutions only take effect because the consuming function imports the
name when it runs, rather than binding it once at module import. Moving those
imports to the top of `helpers.py` silently defeats the substitution and the
tests fail against the real implementation.

Those two statements therefore stay where they are, now with a comment saying
why. `detector_noise` and `detector_rates` came from one local import
statement; `detector_noise` is hoisted and `detector_rates` stays local, which
looks asymmetric but is the point.

## Verification of the import changes

- Every hoisted name is the identical object the local import produced
  (checked by `is`), so no call site resolves to anything different.
- The suite's failing set is byte-identical before and after, isolated by
  stashing only the touched files. The count moves between 20 and 22 depending
  on whether the working tree has uncommitted notebook edits, which
  `tests/test_notebook_noise.py` reads; that is unrelated to this branch.

## Verification

- `tests/test_xdisp_geometry.py`: 2 failed, 11 passed. The two failures are
  pre-existing on `main` (a float `samples` reaching `np.linspace`), unrelated
  to this branch; `main` passes 10, so the extra pass is the new test.
- ZShooter `SPEC` fingerprint with the matching ScopeSim branch applied —
  trace catalog, detector geometry, background budget at both slit widths,
  trace resolution, transmission and emissivity, QE accounting, slit-loss,
  science-truth crosscheck, six image planes and six readouts:
  **byte-identical to `main`**.
