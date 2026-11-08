# Feature ablation: what predicts chess game outcomes?

Each factor (board, skill, time, history) was added one at a time to the same network, trained on the same positions, to measure how much it adds to predicting the game result (white win / draw / black win). This builds on Anderson, Kleinberg and Mullainathan (KDD 2016), who studied skill, time and difficulty using hand-built features on endgames only. Here the features are learned, from positions across whole games.

![Test loss by feature set and validation loss curves](ablation.png)

## Results

Test set, mean ± std over 3 seeds. Loss is cross-entropy, so lower is better.

| Features | Test loss | Accuracy | Change vs board | Change vs previous |
|---|---|---|---|---|
| board | 0.7465 ± 0.0002 | 61.6% ± 0.1 | — | — |
| + skill | 0.7357 ± 0.0003 | 62.6% ± 0.0 | −0.0108 | **−0.0108** |
| + time | 0.7060 ± 0.0007 | 64.6% ± 0.1 | −0.0406 | **−0.0297** |
| + history | 0.7064 ± 0.0002 | 64.5% ± 0.0 | −0.0401 | +0.0004 |

Baselines on the same test set:

- Always predict the most common result: 49.3%
- Predict that the higher-rated player wins: 52.4%

Recall by outcome (mean over seeds):

| Features | White win | Draw | Black win |
|---|---|---|---|
| board | 70.7% | 6.8% | 57.2% |
| + skill | 70.5% | 7.6% | 59.6% |
| + time | 72.2% | 9.2% | 61.9% |
| + history | 72.3% | 9.8% | 61.5% |

## Findings

- **The board alone does a lot.** The board-only model gets 61.6%, well above both baselines, so it learns real signal from the position.
- **Skill helps a little.** Adding both players' Elo lowers the loss by 0.011.
- **Time helps the most.** Clocks and time control lower the loss by about 0.030 on top of skill, nearly three times the gain from skill. Part of this is likely the flagging effect: a nearly empty clock reveals who will lose on time, since "Time forfeit" games are included.
- **History adds nothing measurable.** The change (+0.0004) is within seed noise, and the validation curves show the history model never learned faster than the one without it. Possible reasons:
  - The current position may already hold almost everything the last 8 positions say about the outcome.
  - The encoding may not suit the network: 96 raw planes fed straight into the first layer, trained for only 5 epochs.
- **The ranking is reliable.** Seeds agree within 0.0007, while the gaps between configs are 15–40 times larger. History is the exception, as its gap is within the noise.
- **Draws are rarely predicted.** Draw recall stays below 10% in every config. Draws are rare, so predicting them seldom pays off for the model.

## Caveats

- **Order matters.** The ablation is cumulative, so "time" is measured after "skill". Skill and time overlap (stronger players tend to manage their clocks better), so each factor alone could show a different gain.
- **Different target from the paper.** Anderson et al. predicted errors (blunders) in tablebase-verified endgames. This predicts the final game result from positions across the whole game.

## Next steps

1. **Separate time pressure from flagging.** Rerun with "Time forfeit" games excluded, or with positions taken further from the end of the game.
2. **Measure each factor on its own:** `--configs board+skill board+time board+history`.
3. **Give history a fairer test.** Encode it as the last few moves instead of full boards, or train for longer.

## Setup

- **Data:** Lichess rated standard games, August 2026, via `extract.py --max-games 1750000`. 3.77M games were parsed and 1.75M kept.
  - Filters: both players ≥ 1200 Elo, no bullet (base + 40 × increment ≥ 180 s), terminations "Normal" or "Time forfeit" only.
  - Sampling: 4 random positions per game, skipping the first 8 half-moves and the final position.
  - Splits are assigned per game (no game appears in two splits): 6,287,412 train, 348,629 val and 349,407 test positions.
- **Model:** ResNet with 6 residual blocks of 128 channels on the 8×8 board. Skill and time are fed to the head; history adds 12 planes per earlier position (8 positions, 96 planes).
  - Parameters: 2.32M (board) to 2.44M (with history).
- **Training:** 5 epochs, batch size 512, AdamW (lr 2e-3, weight decay 1e-4) with a OneCycle schedule, mixed precision on an RTX 3070 Ti. The checkpoint with the best validation loss is the one tested.
- **Seeds:** 0, 1, 2.
- **Command:** `python ablate.py --data data --epochs 5 --seeds 0 1 2 --out ablations_v2`

## Files in this folder

- `results.csv`: one row per run, including validation loss curves
- `summary.csv`: the averaged table above
- `ablation.png`: the plot above
- `<features>_s<seed>.pt` and `<features>_s<seed>_loss.png`: checkpoint and loss curve for each run
