# Study workflow

All experimental code lives under `study/`.

## Eval-set workflow

1. Build a candidate manifest from existing held-out physical annotations:

   ```bash
   ./.venv/bin/python study/eval/build_candidates.py
   ```

   This writes `study/eval/candidates.jsonl`.

2. Auto-select the eval breakdown:

   ```bash
   ./.venv/bin/python study/eval/auto_select.py
   ```

   This writes `study/eval/selection.jsonl` and `study/eval/selection_summary.json`.

   If needed, hand-edit `study/eval/selection.jsonl` afterward.

3. Materialize the final eval set:

   ```bash
   ./.venv/bin/python study/eval/materialize_selection.py
   ```

   This writes:

   - `study/eval/frames/*.jpg`
   - `study/eval/labels.jsonl`

## Study 1: base-head

Train:

```bash
./.venv/bin/python study/base-head/train.py --source replay --max-rows 50000
```

Eval:

```bash
./.venv/bin/python study/base-head/eval.py --checkpoint <path/to/base_head.pt>
./.venv/bin/python study/base-head/eval.py --checkpoint <path/to/base_head.pt> --mask
```

## Study 2: minimal DETR

Train:

```bash
./.venv/bin/python study/detr-minimal/train.py --source replay --max-rows 50000
```

Train RT-DETR on the same study setup:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python study/detr-minimal/train.py --architecture rt-detr --source replay --max-rows 50000
```

On this Mac/MPS environment RT-DETR backprop currently needs PyTorch's MPS CPU fallback because
`aten::grid_sampler_2d_backward` is not implemented natively on MPS in the installed torch build.

Eval:

```bash
./.venv/bin/python study/detr-minimal/eval.py --checkpoint <path/to/detr_minimal.pt>
```

Eval an RT-DETR checkpoint:

```bash
./.venv/bin/python study/detr-minimal/eval.py --checkpoint <path/to/rt_detr.pt>
```

## Study 3: geometric mask

No training. The mask is applied through Study 1 eval with `--mask`.

## Final readout

- `study/base-head/RESULTS.md`
- `study/geometric-mask/RESULTS.md`
- `study/detr-minimal/RESULTS.md`
- `study/FINAL_DECISION.md`
