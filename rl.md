# Future RL & Continuous Pre-Training Ideas

If supervised training alone doesn't yield sufficient accuracy, these are directions to explore.

## Continuous Pre-Training

### Vision encoder domain adaptation

DINOv2 is frozen and was pre-trained on ImageNet-scale natural images, not chess boards. Unfreezing it and training on a self-supervised objective (masked image modeling or DINOv2's own iBOT/DINO loss) using a large corpus of unlabeled chess board images would shift the patch embeddings from generic visual features to chess-board-aware features — better encoding of piece shapes, square colors, occlusion patterns, and lighting conditions — before the supervised phase begins. Unlimited data available via Blender + YouTube pipeline frames.

### Temporal module pre-training on raw game sequences

Mamba-2 currently learns chess dynamics jointly with vision during supervised training. Pre-training it on millions of PGN games (Lichess database has ~5B games) as a pure sequence prediction task — given moves 1..t, predict move t+1 (with legal mask) — would give the temporal module strong priors about chess game flow (openings, tactical patterns, endgame transitions) before it ever sees a video frame. No vision needed, just move indices. This is likely the cheapest high-impact win.

### Rolling ingestion of new real data

As the YouTube pipeline discovers and processes new tournament footage, continuously update the model on new clips. Key challenge is catastrophic forgetting — requires replay buffers (mix old + new clips) or EWC-style regularization to maintain performance on previously-learned board themes and lighting conditions.

## Reinforcement Learning

### Game-level REINFORCE / PPO

The model is already a policy — it takes observations (video frames) and outputs actions (move predictions) sequentially. The supervised loss treats each frame independently, but RL can optimize for game-level outcomes:

- **Reward**: Run full rollout on a video, produce predicted PGN, compare to reference PGN.
  - `r = prefix_accuracy(predicted, reference)` — how many consecutive correct moves before the first error.
  - `r = 1 - normalized_edit_distance(predicted, reference)`
- **Why this helps**: Supervised CE loss doesn't penalize error cascading. If the model gets move 12 wrong, all subsequent legal masks shift, but CE still treats each frame independently. RL with game-level reward explicitly teaches the model that early errors compound.

### Stockfish / Leela as a reward model

For unlabeled video where ground truth PGN is unavailable:

- Run inference to get predicted move at position P.
- Query Stockfish: is this move among the top-5 engine moves for position P?
- Reward = 1 if yes, scaled penalty if no (proportional to eval drop).
- This unlocks training on any chess video without ground truth annotations — the engine acts as a preference oracle.
- Especially powerful for self-training: scrape tournament videos, run inference, use engine-validated predictions as pseudo-labels.

### DPO / preference optimization via beam search

Beam search already exists in `MultiGameTracker`. Natural extension:

- For each video, generate K candidate PGNs via beam search.
- Rank by PGN quality (edit distance to reference, or engine evaluation).
- Train with DPO: preferred PGN (closer to reference) vs rejected PGN (further).
- No reward model needed — just pairwise comparisons between the model's own outputs.

### Detection threshold RL

The move detection head outputs a binary signal, currently trained with focal loss. RL alternative:

- Reward: +1 for correct detection, -1 for false positive (hallucinated move), -0.5 for false negative (missed move).
- The asymmetry matters — a false positive inserts a wrong move and corrupts all subsequent state. Supervised focal loss doesn't capture this asymmetry as well as shaped rewards.

### Online self-improvement loop

The full loop:

1. Scrape new tournament videos (pipeline/crawl).
2. Run inference to produce predicted PGNs.
3. Score predictions via engine validation (Stockfish top-5 agreement), game coherence (reasonable length, proper termination), and high-confidence filtering.
4. Add high-scoring predictions as new training data (pseudo-labels).
5. Fine-tune model with a mix of original supervised data (replay buffer), new pseudo-labeled data, and RL loss on game-level reward.
6. Evaluate on held-out benchmark.
7. If improved, checkpoint becomes new base model. Repeat.

## Priority ranking

| Approach | Effort | Expected Impact | Data Needed |
|---|---|---|---|
| Temporal pre-training on PGN | Low | High | Free (Lichess DB) |
| Vision CPT on board images | Medium | Medium | Already have (Blender + YouTube) |
| Game-level REINFORCE | Medium | High | Existing labeled clips |
| Stockfish reward on unlabeled video | Medium | Very high | Unlabeled YouTube (unlimited) |
| DPO via beam search | Low | Medium | Existing labeled clips |
| Full self-improvement loop | High | Very high | Pipeline + engine |
