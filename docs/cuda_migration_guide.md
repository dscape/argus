# CUDA migration guide — argus classifier experiments

**Status:** followup doc. Nothing to do today unless a classifier experiment shows non-deterministic behavior that a CUDA run could settle.

## Why move to CUDA

The April 2026 classifier-experiment round observed two MPS-specific failure modes on the M-series Mac dev box:

1. **Non-deterministic training outcomes.** Training DINOv3 piece classifier with the exact same config gave `best_val_accuracy = 0.748` in one run and `0.462` in another. No source-level change between the two — same seed, same batch size, same dataset. MPS ops in `torch` occasionally produce non-deterministic results across runs (there is no reliable equivalent of `torch.use_deterministic_algorithms(True)` on MPS as of torch 2.5).
2. **`torch.save` hang on `.item()` sync.** When `best_state` contained DINOv3 tensors still materialized on MPS (not `.cpu()`-copied), `torch.save` deadlocked for 10+ minutes on a `Tensor.item()` GPU→CPU sync inside the serializer. Workaround shipped in [scripts/train_square_classifier.py](../scripts/train_square_classifier.py): always `v.detach().cpu().clone()` the best_state before saving.

Neither failure mode has been reproduced on CUDA (because we haven't tried). The CUDA migration exists to:
- Settle whether DINOv3 is a real improvement (need reproducible training runs).
- Re-run the MPS-flaky studies (3c, DINOv3) to separate "MPS bug" from "ineffective intervention."
- Unlock the broadcast-synth pipeline at scale (rendering + training on a beefier GPU).

## What you need on the CUDA host

### Hardware

- Single GPU, 24GB VRAM recommended (A5000 / L4 / 3090 all fine). DINOv2-base + piece head trains in ~15 min on a 3090. Synth rendering is the heavier workload — that should run on a separate Blender-capable box anyway.
- 50GB free disk for code + data + checkpoints + HF cache.
- ~32GB RAM.

### Software

- **Python 3.10 or 3.11.** Argus [pyproject.toml](../pyproject.toml) requires `>=3.10`. `recap==0.1.6` (chesscog's config dep that we've been running) requires `<3.11` per its pyproject, though we have a working shim at `.venv/lib/python3.14/site-packages/recap/path_manager.py` that handles Python 3.12+. On a fresh CUDA box, **just use Python 3.10** to avoid the shim.
- **CUDA 12.1+** with compatible driver. Match to the torch wheel you install.
- **PyTorch ≥ 2.2** for `torch.load(..., weights_only=False)` default behavior that the rest of the repo uses. Any modern CUDA-capable torch works.
- **HuggingFace transformers ≥ 5.0** for `DINOv3ViTModel`. Current wheel in `.venv/` is `transformers==5.3.0`.

### Dataset staging

Argus data directories to ship (approximate sizes on Apr 19, 2026):

| Path | Size | Required for |
|---|---:|---|
| `data/physical/` | 466MB | classifier training + eval |
| `data/argus/` | 2.7GB | synthetic clips (only needed if you're training on synth) |
| `data/chesscog_baseline/` | 1.4GB | chesscog PNG dataset (regeneratable from physical + chesscog scripts) |
| `data/study1_argus_geom/` | 1.3GB | Study 1 dump (regeneratable) |

The `data/physical/` dataset references native video files via `source_video_id` + `source_frame_index`. `NativeFrameLoader` at [pipeline/physical/two_stage/classifier_data.py](../pipeline/physical/two_stage/classifier_data.py) uses `resolve_source_video_path()` (in `pipeline/physical/shared/source_video_paths.py`) to locate them. Make sure those videos are also on the CUDA host at the same absolute path, OR update `resolve_source_video_path()` to point at their new location.

### HuggingFace login

DINOv3 checkpoints are gated. Log in to HF on the CUDA box with an account that has accepted the license on each DINOv3 repo you need:

```bash
hf auth login  # paste a read-only token
```

Verify:

```bash
.venv/bin/python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').model_type)"
# Expected output: dinov3_vit
```

## Mac-specific gotchas to check on Linux

The Apr-19 MPS environment had two non-fatal warnings that are worth confirming don't appear on Linux:

1. **cv2 + av dylib conflict** (`AVFFrameReceiver`, `AVFAudioReceiver` both implemented in two libs). Linux uses ffmpeg-native builds; this warning should be absent but test for it.
2. **MPS fallback ops.** Some torch ops fall back to CPU on MPS silently and produce different numerical results. CUDA runs all ops on GPU; comparing training loss curves between Mac and Linux may show small differences due to this.

## Minimal smoke test

After dataset staging and `hf auth login`, verify the baseline reproduces:

```bash
# 1. Install deps (match your CUDA version)
.venv/bin/pip install torch torchvision transformers>=5.0 -U

# 2. Reproduce the Apr-17 baseline on 10 val boards
.venv/bin/python scripts/eval_two_stage_board_reader.py --device cuda --limit 10 \
  --occupancy-checkpoint weights/physical/square_classifier/occupancy/occupancy_classifier.pt \
  --piece-checkpoint weights/physical/square_classifier/piece_corrected/piece_classifier.pt

# Expected: per-square ≈ 0.77, non-empty ≈ 0.59, board_exact = 0 or 1 out of 10
```

If the numbers are meaningfully different (±2pp), there's a device-specific divergence worth investigating before running any new training.

## DINOv3 reproducibility test

The key experiment to run first on CUDA:

```bash
# Run 3 times with identical args, compare best_val_accuracy variance
for i in 1 2 3; do
  .venv/bin/python scripts/train_square_classifier.py \
    --task piece --encoder-type dinov3 --augment \
    --output-dir weights/debug/dinov3_run_$i \
    --epochs 4 --batch-size 32 --device cuda --seed 42
done
```

Then inspect the three `summary.json` files. Interpretation:

- **Variance < 1pp:** MPS was the source of non-determinism. DINOv3 is a real alternative; rerun the piece training at best settings and evaluate end-to-end. If end-to-end still doesn't translate the classifier-val gain, it's the "classifier-val vs end-to-end" issue (see Phase B diagnostic) rather than the backbone.
- **Variance ≈ 20pp (matching MPS):** the non-determinism is not device-specific. DINOv3 training is itself unstable — likely an AdamW + register-token interaction. Workaround: use a lower learning rate (try `--lr 1e-4` instead of default 3e-4) or increase warmup.

## What to migrate vs keep on Mac

| Task | Where it should run |
|---|---|
| Classifier training (DINOv2, DINOv3, ResNet18) | CUDA |
| Classifier evaluation (`eval_two_stage_*`, `compare_*`, `eval_mixed_stack`, `study*`) | CUDA |
| Diagnostic scripts (`diagnose_classifier_val`) | CUDA (reuses the same models) |
| Broadcast synth rendering (Blender) | Separate render box or local Mac (one-time cost) |
| Dev-tools UI (`dev-tools/`) | Local Mac (interactive, not compute-intensive) |
| Data exploration / paper reading | Local Mac |

## Open questions to settle once on CUDA

1. **DINOv3 reproducibility** (see above).
2. **3c unfreeze end-to-end**: did the classifier-val +1.3pp fail to translate because of MPS noise, or is the pattern real? Rerun 3c, evaluate end-to-end.
3. **Phase A mixed stack**: rerun on CUDA to confirm findings.
4. **Synth data pipeline**: the main future-work item; deferred until rendering story is sorted.

## Fallback: if you don't have a CUDA host

The current Mac dev environment still produces consistent DINOv2 results — only DINOv3 was flaky. For now, **stick with DINOv2** on Mac for anything that needs to be reliable. DINOv3 experiments should be gated behind the CUDA migration.
