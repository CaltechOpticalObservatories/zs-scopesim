
Use conda run -n zssim jupyter server list to see what might be active or executing before getting started.

This repository favors transparent, directly inspectable scientific code over generalized library-style abstractions.

### ACP memory guard
- heap limit is about 4 GiB and distinct from IDE's overall 8GiB heap limit 
- Do not print, serialize, or send entire notebook files, notebook outputs, or full multi-thousand-line diffs through the chat/tool stream. 
- Inspect notebook source by relevant cell IDs or narrow source ranges. Avoid loading large notebook output blobs into the conversation.

### Directness and readability

- Write the physical calculation where it is used, in the same order that a scientist would derive or inspect it.
- Prefer direct access to known configured objects, tables, metadata, and quantities.
- Do not wrap a short calculation in single-use helpers.
- Introduce a helper only when it is reused, independently meaningful, or substantially clarifies the calling code.
- Avoid policy layers, schemas, adapters, contracts, registries, audit structures, and speculative abstractions unless explicitly requested.
- Optimize for rapid inspection of the calculation, not minimum line length or maximum API generality.

### Failure behavior

- Let ordinary `KeyError`, `AttributeError`, unit errors, shape errors, and indexing errors propagate when they already identify the broken assumption.
- Do not wrap every operation in custom validation and verbose exceptions.
- Do not use `try`/`except`, `getattr`, `.get`, `nan*` operations, coercions, or default values to conceal missing or inconsistent physical configuration.
- Never silently substitute conventional values such as Nyquist sampling, two pixels, median parameters, or generic defaults.

### Sources of truth

- Use configured simulation or data objects as the authoritative source.
- Do not independently recreate geometry, transformations, dispersion, sampling, or other quantities already represented by the configured model.
- Heavily prefer values and methods from the object that actually performs the simulation.
- Preserve distinctions between coordinate systems. Internal model coordinates, physical focal-plane coordinates, image-plane coordinates, and detector pixels must not be relabeled as one another.
- If an internal/private attribute is intentionally the best representation of the configured model, use it directly. Do not add indirection merely because it is private.

### Units, arrays, and returned types

- Inspect actual returned types, units, and shapes before writing conversions.
- Preserve `Quantity` objects until a plain numeric column or plotting API requires values.
- Avoid unnecessary `float`, `np.asarray`, `u.Quantity`, `np.nanmean`, and similar defensive conversions.
- Do not average or flatten values unless the averaging operation has an explicit physical meaning.
- Keep equations short enough that their units and physical interpretation remain visible.

### Tables and derived data

- Include columns that represent real scientific quantities or actual downstream needs.
- Do not add provenance, compatibility, status, note, or diagnostic columns by default.
- Locate consumers before adding compatibility aliases.
- Do not retain physically incorrect columns for backward compatibility.
- Represent configured constants as single values, not artificial min/median/max ranges.
- Retain ranges only for quantities that genuinely vary and are scientifically useful.

### Plotting

- Preserve existing plot semantics, layout, titles, annotations, and visual density unless changing them is explicitly in scope.
- Do not change an unrelated title or annotation merely because the underlying table schema changed.
- Plot existing model geometry directly when available.
- Do not introduce a second geometry model solely for visualization.
- Avoid coarse subsampling, excessive linewidths, per-sample strips, or rendering choices that introduce aliasing, seams, or apparent discontinuities.
- When coloring a continuous trace, connect adjacent samples with ordinary continuous segments and inspect the rendered result.
- Treat visual inspection as necessary for plot changes; passing a smoke test is not sufficient.

### Formatting and comments

- Do not mechanically reformat surrounding code.
- Do not optimize code for Black-style wrapping or a formatter’s preferred appearance.
- Prefer compact expressions when they make the scientific relationship easier to see.
- Break lines where it helps the reader understand the calculation, not merely to satisfy a nominal line length.
- Comments and docstrings should explain non-obvious physics, units, coordinate conventions, or deliberate approximations.
- Do not narrate obvious control flow or restate the code in prose.
- One-line private helpers and notebook utilities generally do not need elaborate docstrings.
- When LaTeX is supported (especially Matplotlib/Jupyter labels), write symbols with ASCII-only LaTeX/mathtext such as `r"$\Delta m_5$"` and `r"$R\approx1000$"`; never insert literal Unicode mathematical or Greek characters into source.

### Scope discipline

- Treat dirty edits as evidence of intent, not automatically as correct implementation.
- Preserve unrelated user changes.
- Suggest broadening a correction into cleanup of neighboring systems but require explicit approval to execute.
- When a discovered issue is real but outside scope, report it separately instead of folding it into the current patch.
- Before changing a public name, table column, plot, or notebook output, locate its actual consumers.

### Testing and review

- Test the scientific relationship, not incidental implementation structure.
- Use targeted numerical tests with physically interpretable values.
- Test that required missing configuration fails naturally and that no fallback result is manufactured.
- Avoid misleadingly precise tolerances unless justified by the numerical method and model.
- For plots, test basic structure programmatically and inspect a rendering using the real configuration.
- After implementation, review the complete touched path and delete scaffolding, redundant conversions, unused helpers, and obsolete columns.
- A patch should normally leave the relevant calculation shorter and easier to audit than it found it.
