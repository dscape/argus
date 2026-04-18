# Piece-projection refactor plan

## Status

- Phase 0 complete: tagged `pre-cleanup-archive` from `main` HEAD and branched to `refactor/piece_projection`.
- Phase 1 discovery complete. The findings below are the deletion gate for the rest of the refactor.
- Phase 2 complete: extracted `resolve_source_video_path` to `pipeline/physical/shared/source_video_paths.py`, removed the safe dead scripts/modules/tests, and deleted the whole-board lane.
- Phase 3 complete: expanded `pipeline/physical/piece_projection.py` with board-neighborhood crop and per-square base-quad/bbox helpers plus unit coverage.
- Phase 4 complete: board-probe runtime/training, failure-study tooling, and move-model data all dropped the deleted oblique/native-oblique geometry paths; the joint-reader bridge was removed; dead geometry modules were deleted.
- Phase 5 complete: annotation-row loading moved to `pipeline/physical/shared/annotation_rows.py` and live importers were repointed there.
- Phase 6 complete: `scripts/prepare_square_classifier_dataset.py` now uses projected crops directly.
- Phase 7 complete: `square_data.py` was decomposed away and square-class constants now come from shared modules.
- Phase 8 complete: `square_crop.py` was folded away; two-stage crop-size constants now live in `pipeline/physical/two_stage/classifier_data.py`.
- Phase 9 complete: `pipeline/physical/` was reorganized into `board_probe/`, `two_stage/`, and `shared/` with all live imports updated.
- Phase 10 complete: the board-probe and two-stage classifiers were retrained or warm-start retrained, and the resulting metrics meet or exceed the required baselines.
- Phase 11 complete: autoresearch cache was regenerated from the refactored checkpoint, the top decoder group was re-evaluated, and decoder **v282** was promoted as the default production decoder.
- Phase 12 complete: README and `progress.md` were updated to reflect the real physical architecture and remove stale references to deleted files.
- Verification after Phases 2-12: `.venv/bin/pytest tests/`, `make lint`, and `make typecheck` are green; runtime/dev-tools physical routes were smoke-tested locally against the new outputs.

## Phase 1 discovery findings

### Gate 1 — board-probe geometry touchpoints

#### `pipeline/physical/board_probe.py`

- `dino_patches_to_square_tokens(...)`
  - Current behavior: assumes the encoder saw an already rectified square board image, reshapes the patch grid, and averages uniform `grid/8 x grid/8` patch blocks into 64 square tokens.
  - Geometry status: not a geometry helper; this is board-probe pooling logic.
  - Refactor implication: it can survive only where the image fed to it is already piece-projection-normalized to a square board view. It should not stay coupled to rectified legacy helpers.
- `sample_oblique_square_tokens_from_patch_tokens(...)`
  - Current behavior: computes an image→board homography with `cv2.getPerspectiveTransform`, maps patch centers into board coordinates, then applies hand-tuned row/file margins to pool contextual tokens per square.
  - Geometry status: this is oblique geometry plus board-probe pooling fused into one function.
  - Refactor implication: this function should be deleted. If board-probe still pools from a board-neighborhood image, the pooling logic should stay in `board_probe/`, but its geometry inputs must come from `piece_projection.py` rather than ad-hoc homography math here.
- `extract_square_token_features(...)`
  - Current behavior: dispatches between three geometry contracts based on dataset shape:
    - `(images, labels)` with per-square images -> no board geometry, encoder pooled per square.
    - `(images, labels)` with whole-board images -> `dino_patches_to_square_tokens(...)`.
    - `(images, labels, corners)` -> `sample_oblique_square_tokens_from_patch_tokens(...)`.
  - Refactor implication: after the cleanup there should be one surviving whole-board geometry contract, sourced by `piece_projection.py`.

#### `pipeline/physical/square_classifier.py`

- `_predict_board_logits(...)`
  - `architecture in {"board_probe", "board_probe_ensemble"}`:
    - `board_input_mode == "oblique_square_context"`:
      - Uses `extract_oblique_square_context_crops(...)`.
      - Dead under the new plan.
    - `board_input_mode == "oblique_board"`:
      - Uses `preprocess_oblique_board_image(...)` for board cropping + corner rescaling.
      - Then uses `sample_oblique_square_tokens_from_patch_tokens(...)`.
      - Dead as written; any surviving board-neighborhood path must get crop geometry from `piece_projection.py` and pooling from `board_probe/`.
    - `board_input_mode == "oblique_board_crop"`:
      - Uses `preprocess_oblique_board_image(...)` and then uniform square-grid pooling.
      - Dead as written for the same reason.
    - `board_input_mode == "rectified_board"`:
      - Assumes callers already handed it a rectified board image and uses uniform square-grid pooling.
      - This is the current production/default path.
  - `architecture == "square_probe"`:
    - `oblique_square_context` uses heuristic context crops.
    - Otherwise uses `split_rectified_board_into_squares(...)`.
- `_predict_board_logits_batch(...)`
  - Fast batched path exists only for:
    - board-probe + `rectified_board`
    - square-probe + non-oblique-square-context
  - All oblique modes fall back to per-frame inference.
  - Refactor implication: once one geometry contract remains, the batch path should be simplified around that contract.

#### `scripts/train_physical_board_probe.py`

The script itself does not implement geometry directly; it selects dataset families that do:

- `board_input_mode == "rectified_board"`
  - Synthetic: `PhysicalSyntheticClipBoardDataset`
  - Eval/manual: `PhysicalEvalBoardDataset`, `PhysicalManualTrainBoardDataset`
  - Real pseudo-real: `PhysicalRealBoardDataset`
  - These all depend on legacy rectified-board geometry (`rectify_board_image(...)`, pre-saved rectified boards, or rectified replay exports).
- `board_input_mode == "oblique_square_context"`
  - Uses `Physical*ObliqueSquareContextDataset`
  - Depends on `extract_oblique_square_context_crops(...)`.
- `board_input_mode == "oblique_board"`
  - Uses `Physical*ObliqueBoardDataset`
  - Depends on `extract_oblique_board_crop(...)` + scaled corners.
- `board_input_mode == "oblique_board_crop"`
  - Same crop path, then drops corners.

Refactor implication: the mode surface should collapse to the single surviving piece-projection-native board-probe input contract.

#### What `pipeline/physical/piece_projection.py` already provides

- `board_to_image_homography(...)`
- `camera_pose_from_corners(...)`
- `project_points(...)`
- `project_points_with_base_homography(...)`
- `project_piece_box(...)`
- `piece_bbox_from_projection(...)`
- `extract_projected_piece_crop(...)`
- `extract_projected_occupancy_crop(...)`

#### What `pipeline/physical/piece_projection.py` still needs to gain

- A reusable board-neighborhood crop helper to replace `extract_oblique_board_crop(...)` / `preprocess_oblique_board_image(...)`, including relative-corner remapping after the crop.
- Reusable per-square planar quad / bbox helpers so callers do not re-derive square geometry with their own homography math.
- Reusable full-board / per-square bbox extraction for board-probe patch pooling.

It does **not** need to absorb patch→square token pooling, image normalization, or model/input constants.

### Gate 2 — `move_data.py` oblique callers

Live non-test consumers of `pipeline/physical/move_data.py` are:

- `scripts/train_physical_move_model.py`
- `scripts/eval_physical_move_model.py`

Inside `move_data.py`, the non-rectified geometry is concentrated in:

- imports:
  - `from pipeline.physical.oblique_board_data import extract_oblique_board_crop`
  - `from pipeline.physical.oblique_square_context import _load_clip_frame_bgr, load_annotated_oblique_rows`
- mode surface:
  - `observation_mode in {"rectified", "oblique", "native_oblique"}` across
    - `build_real_move_window_clips(...)`
    - `load_real_move_sequences(...)`
    - `load_eval_move_sequences(...)`
- helpers:
  - `_prepare_oblique_clip_frame(...)`
  - `_prepare_native_annotation_oblique_frame(...)`
  - `_prepare_native_real_oblique_frame(...)`

Conclusion:

- After the dead whole-board / oblique-reader deletions, there is no production caller that needs the `oblique` or `native_oblique` move-data paths.
- `load_annotated_oblique_rows(...)` and `_load_clip_frame_bgr(...)` still have other live repo consumers, but those consumers are annotation/review/two-stage utilities, not move-model requirements.
- Therefore the move-model oblique branches should be removed rather than ported.

### Gate 3 — autoresearch probe-output contract

`autoresearch/prepare.py` currently caches **runtime square logits**, not probe embeddings.

- It loads held-out rows via `PhysicalEvalBoardDataset()`.
- For each clip it calls `read_board_logits_batch_from_frames(...)`.
- Each returned item is one frame of square logits with shape `(64, 13)`.
- It stacks those into `PreparedSequence.logits` with shape `(num_frames, 64, 13)`.
- `PreparedSequence.target_labels` has shape `(num_frames, 64)`.
- `autoresearch/train.py` and `prepare.evaluate_decoder(...)` assume:
  - 64 squares
  - row-major ordering
  - 13-way square logits per square

Implications:

- If the refactor preserves the `64 x 13` row-major board-logit contract, `autoresearch/` only needs import/runtime-path updates plus cache regeneration.
- If square count/order/class count changes, `autoresearch/prepare.py`, `autoresearch/train.py`, and downstream decoder evaluation will all need code changes.
- Cache regeneration is mandatory after the retrained probe lands even if the tensor shape stays the same, because the weights path and logits will change.

### Gate 4 — promote-which-config

Current `autoresearch/results.tsv` confirms the user’s provisional read:

- The top board-exact cluster is tied at `board_exact=0.254438`.
- `v273` is the recall-favoring member of that cluster:
  - `non_empty_accuracy=0.889670`
  - `macro_f1=0.859107`
  - `move_recall=0.500000`
  - `false_change_rate=0.016818`
- `v282` / `v284` are lower-false-change alternatives:
  - `move_recall=0.484375`
  - `false_change_rate=0.015524`
  - materially worse non-empty / macro than `v273`

Decision:

- Keep `v273` as the **provisional** favorite.
- Do **not** wire any decoder default yet.
- Re-sweep against regenerated post-refactor logits after retraining, then promote the winner.

### Gate 5 — joint-reader / move-model entanglement

#### (a) Symbols imported by `scripts/train_physical_move_model.py`

The train script imports exactly two symbols from `pipeline/physical/joint_board_reader.py`:

- `argus_overrides_from_joint_board_reader_checkpoint`
- `argus_square_reader_state_dict_from_joint_board_reader_checkpoint`

#### (b) Oblique / native-oblique paths in `pipeline/physical/move_data.py`

The move-model data path still supports:

- `observation_mode="oblique"`
- `observation_mode="native_oblique"`

through the helpers listed in Gate 2 and by populating `board_corners` only for those modes.

#### (c) Are the checkpoint-translation helpers generalisable?

Not really.

- `argus_overrides_from_joint_board_reader_checkpoint(...)` hardcodes `square_token_mode="oblique_square_queries"` and maps a joint-reader checkpoint schema into `ArgusModel` kwargs.
- `argus_square_reader_state_dict_from_joint_board_reader_checkpoint(...)` hardcodes a rename from `square_decoder.*` to `square_tokenizer.*` and passes through `square_head.*`.
- Both helpers are specific to the deleted oblique joint-reader family.

Decision:

- Kill the oblique/native-oblique move-model modes instead of preserving them.
- Remove the joint-reader checkpoint-initialization surface from the move-model script(s) rather than extracting those helpers to shared code.
- This is the clean route that lets `pipeline/physical/joint_board_reader.py` die without leaving an archived oblique-square-query initialization path wired into live training.

Follow-on work implied by this decision:

- `scripts/train_physical_move_model.py`
  - remove oblique/native-oblique observation modes
  - remove joint-reader checkpoint initialization
  - update defaults so the surviving configuration is internally consistent
- `scripts/eval_physical_move_model.py`
  - drop removed observation modes and any assumptions that `board_corners` exist
- `pipeline/physical/move_data.py`
  - delete oblique/native-oblique branches and helpers once the scripts above stop using them
- tests
  - delete/update move-data and move-model tests that only exercise the removed oblique surfaces
  - strip any `tests/test_argus_model.py` sections that only validate the joint-reader bridge

## Remaining execution order

1. Safe deletions.
2. Expand `piece_projection.py` only with the missing image-geometry helpers.
3. Purge oblique + rectified geometry branches.
4. Extract shared annotation-loader code.
5. Finish the two-stage migration.
6. Decompose and delete `square_data.py`.
7. Fold and delete `square_crop.py`.
8. Reorganize `pipeline/physical/` into `board_probe/`, `two_stage/`, and `shared/`.
9. Retrain probe + two-stage models.
10. Regenerate autoresearch cache, re-sweep, and promote the winning decoder.
11. Update `README.md` and purge stale `progress.md` references.
