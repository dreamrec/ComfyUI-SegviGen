# ComfyUI-SegviGen V3 Development Roadmap

**Target File**

`docs/superpowers/plans/2026-04-02-upstream-faithful-development-roadmap.md`

**Summary**

This roadmap assumes the current remote baseline is commit `93f40f7` on April 2, 2026, where the repo has already recovered much of the interactive path by introducing a TRELLIS stage shim, real `tex_slat` sampling for the bridge path, packed multi-click interactive inference, and better test scaffolding. The next goal is to finish the job: make the repo upstream-faithful for the original SegviGen paper while preserving the current ComfyUI usability.

The long-term product direction is:

1. Treat **asset-native SegviGen** as the primary architecture.
2. Keep the current **TRELLIS2 ImageToShape bridge** as a supported ComfyUI workflow.
3. Make **interactive segmentation** the first production-complete path.
4. Then replace the current **full segmentation** heuristic path with true upstream behavior.
5. Then add the missing **2D-guided full segmentation** path.
6. Keep all legacy shortcuts explicit, isolated, and non-default.

## 1. Current State Snapshot

1. The repo is no longer in "prototype only" territory. `93f40f7` fixed several major issues from the previous audit: interactive checkpoint auto-download is wired, interactive workflow uses image conditioning again, `guidance_rescale` is forwarded, and the interactive sampler can use a real `tex_slat` when `SegviGenVoxelEncode` succeeds.

2. The repo is still architecturally split in half. The interactive path is recovering toward upstream, but the full sampler is still the old latent-clustering path in `nodes/nodes_sampler.py`. That means the repo currently has one partially-correct path and one still-incorrect path.

3. The new `SegviGenVoxelEncode` is a **bridge encoder**, not a true upstream asset encoder. It samples `tex_slat` from TRELLIS stages using `shape_result + conditioning`; it does not yet perform the original `process_glb_to_vxz -> vxz_to_latent_slat` asset encoding stack from the paper.

4. Export and preview are still built around label grids mapped back to the mesh, not around a fully faithful decoded SegviGen output object. That is acceptable as a compatibility layer, but it should not remain the only export path.

5. Packaging and documentation are still behind the code. `requirements.txt` is incomplete, `pyproject.toml` still has a placeholder repository URL, and the README still describes the project like an Aero-Ex-style node pack rather than a staged recovery toward original SegviGen.

## 2. Final Architecture To Build Toward

1. The repo should support **two official input stacks**:
   - **Asset-native path**: input mesh or GLB -> `.vxz` generation -> `shape_slat + tex_slat + subs` -> sampler -> decode -> export.
   - **TRELLIS bridge path**: image -> TRELLIS2 conditioning -> TRELLIS2 shape generation -> bridge `tex_slat` sampling -> sampler -> decode -> export.

2. Both paths should converge into the same internal contracts:
   - `SEGVIGEN_VOXEL`: asset preprocess payload or bridge placeholder metadata.
   - `SEGVIGEN_SLAT`: real shape and texture latent state plus decode support metadata.
   - `SEGVIGEN_COND`: normalized segmentation conditioning payload with explicit task mode.
   - `SEGVIGEN_SEG_RESULT`: decoded segmentation result with labels, decoded voxels, and export-ready metadata.

3. All TRELLIS private API usage must be isolated behind wrappers in `core/trellis2_shim.py` and adjacent helper modules. No node or high-level sampler code should call `_sample_tex_slat`, `_decode_tex_slat`, or other private TRELLIS functions directly.

4. The repo should explicitly distinguish:
   - **paper-faithful paths**,
   - **bridge-faithful paths**,
   - **legacy compatibility paths**.

5. The UI must never silently degrade from faithful mode to heuristic mode. Every downgrade must either hard-fail or require an explicit advanced opt-in.

## 3. Contract Freeze Before More Features

1. Add a new `core/contracts.py` module and move all shared dict-shape logic there. It should define constructors and validators for:
   - `build_segvigen_voxel(...)`
   - `build_segvigen_slat(...)`
   - `build_segvigen_cond(...)`
   - `build_segvigen_seg_result(...)`

2. Freeze the `SEGVIGEN_SLAT` schema as:
   - `latent`: backward-compatible alias to `shape_slat`
   - `shape_slat`: required
   - `tex_slat`: optional but required for faithful sampling
   - `subs`: optional but required for faithful decode
   - `voxel`: metadata dict
   - `source`: enum string with values `shape_only`, `bridge_full`, `asset_full`
   - `pipeline_type`: `512`, `1024`, or `1536_cascade`-style normalized label
   - `normalization`: optional dict with `center`, `scale`, `resolution`, and `coord_space`

3. Freeze the `SEGVIGEN_SEG_RESULT` schema as:
   - `latent`: backward-compatible alias to `output_tex_slat`
   - `output_tex_slat`: sampler output in normalized or explicitly tagged form
   - `decoded_tex_voxels`: decoded PBR voxel representation when available
   - `labels`: integer label grid
   - `labels_source`: enum string with values `decoded_binary`, `decoded_color_cluster`, `latent_kmeans_fallback`, `none`
   - `mode`: enum string with values `interactive_binary`, `full`, `full_2d_guided`, `preview_passthrough`
   - `mesh`: original mesh
   - `voxel`: metadata dict
   - `source`: copied from `SEGVIGEN_SLAT`

4. Freeze the label convention as:
   - `0`: empty / background / unassigned
   - `1..N`: actual part labels
   - For paper-faithful interactive mode, use `1 = selected part`, `2 = remainder of object` if the remainder is explicitly represented.

5. Add a versioned contract marker such as `segvigen_contract_version = 3` to every high-level payload so future migrations can be explicit.

## 4. Repository-Level Module Direction

1. Keep `core/trellis2_shim.py`, but extend it into the only legal bridge to TRELLIS stage internals. It should expose stable wrappers like:
   - `load_trellis2_stages()`
   - `ensure_supported_trellis2_revision()`
   - `sample_bridge_tex_slat(...)`
   - `decode_bridge_tex_slat(...)`
   - `run_bridge_conditioning(...)`

2. Keep `core/encode.py`, but rename its role in code and docs to **bridge encode**. It should only handle the TRELLIS bridge path.

3. Add `core/asset_encode.py`. This will be the real upstream asset encoder and should own:
   - texture preprocessing
   - scene normalization to `[-0.5, 0.5]`
   - `o_voxel` flexible dual-grid conversion
   - `.vxz` serialization
   - `vxz_to_latent_slat(...)`
   - shape encoder, tex encoder, shape decoder loading

4. Add `core/decode.py`. This module should own:
   - denormalization of `output_tex_slat`
   - `tex_decoder` or `_decode_tex_slat` calls
   - binary interactive label extraction
   - full color-cluster label extraction
   - conversion from decoded voxel labels to exportable structures

5. Keep `core/sampler.py`, but evolve it into the single source of truth for SegviGen sampling schedule behavior. It must no longer be just a thin forwarding shim once full parity is implemented. It should own:
   - guidance interval logic
   - `guidance_rescale`
   - `rescale_t`
   - shared timestep generation
   - CFG math parity with upstream scripts

6. Split `core/interactive.py` into:
   - point token packing and point-coordinate mapping
   - interactive model wrapper
   - interactive binary decode helpers
   - optional legacy fallbacks only if they remain supported

7. Add `core/checkpoints.py`. Move all checkpoint source definitions and download logic there. `install.py` should become a thin CLI wrapper around this module.

## 5. Development Phases

### Phase 0 — Baseline Freeze And Safety Rails

1. Create the roadmap file above, add a short "state of repo" section to the README, and mark the current full sampler as `experimental` in the display name until Phase 4 is complete.

2. Add `core/contracts.py` and move all payload construction to it. Do this before any new features so the rest of the work is not built on ad hoc dict mutations.

3. Add a TRELLIS revision compatibility check in `core/trellis2_shim.py`. It must validate the presence and callable signatures of the private methods the repo now depends on. If the installed `ComfyUI-TRELLIS2` is incompatible, fail with one actionable message.

4. Change the interactive sampler to hard-error by default when `slat["source"]` is `shape_only`. Keep a single advanced flag such as `allow_legacy_shape_only_fallback = false` if temporary escape hatches are still needed during development.

5. Acceptance criteria:
   - no silent quality downgrade remains in the default interactive path,
   - all payloads carry source and mode metadata,
   - the README and node display names no longer imply that full mode is already faithful when it is not.

### Phase 1-9

See sections below for full details on each phase.
