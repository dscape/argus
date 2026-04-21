# Template-matching results

## Stage 7

Template bank:

- bank: `study/templates/data/h2WrtkfwRl8.pt`
- setup frames: native source-video frames `100, 102, 104, 106`
- eval scope: all eval frames with `source_video_id = h2WrtkfwRl8` (`183` frames)

Hard-category average here is the mean of:

- `lateral-occlusion`
- `low-camera-angle`
- `dense-middlegame`
- `mid-move`

Baselines below come from `study/FINAL_DECISION.md` on the full held-out eval set, so they are not apples-to-apples with the `h2WrtkfwRl8`-only template runs.

| variant | eval scope | overall piece F1 | hard-category avg F1 | lateral-occlusion | low-camera-angle | dense-middlegame | mid-move |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base-head | full eval (`344`) | 0.0014 | 0.0012 | 0.0017 | 0.0018 | 0.0015 | 0.0000 |
| base-head + mask | full eval (`344`) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| minimal DETR | full eval (`344`) | 0.2338 | 0.2539 | 0.2124 | 0.2493 | 0.2813 | 0.2724 |
| templates + cuboid | `h2WrtkfwRl8` only (`183`) | 0.1701 | 0.1382 | 0.1661 | 0.1464 | 0.1837 | 0.0564 |
| templates + SAM3 | `h2WrtkfwRl8` only (`183`) | 0.0085 | 0.0068 | 0.0041 | 0.0060 | 0.0169 | 0.0000 |

## Run artifacts

- cuboid: `study/templates/eval/h2WrtkfwRl8_cuboid/metrics.json`
- sam3: `study/templates/eval/h2WrtkfwRl8_sam3/metrics.json`
