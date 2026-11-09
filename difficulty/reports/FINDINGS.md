# What the difficulty scores show (plain-language write-up)

Data behind this: 1,000 random positions from the **test split** of a small sample (Lichess June 2013, about 153k games
extracted with the default settings), scored with Stockfish 19 (200,000 nodes per position) and the team's current
`model.pt`. Ratings 1300 to 2000 (the range this sample covers well). Positions come from only about 250 games
(4 per game), so they are not fully independent. Treat everything here as a first look, not a result.

Difficulty = Stockfish's expected score for the side to move minus the model's expected score for the side to move
(expected score = win chance + half the draw chance). Positive means humans do worse than the engine says.

## The main finding: difficulty does not change with rating, because the model barely uses rating

* Average difficulty is **+0.042 at 1300 and +0.042 at 2000** (`difficulty_by_rating.png`). The change is 0.0001.
* In the 20-position check (`rating_sweep.md`), the largest change in any one position from 1200 to 2400 was 0.017.
  Winning positions did not get easier to convert for stronger players (+0.0005 on average), and
  losing positions did not get harder.
* This is the model, not a bug in our code: even pushing the ratings to extreme values (0 to 3000) moves predictions by
  under 0.05. The model reacts somewhat more to the **gap** between the two ratings (a 2200 side to move against a 1200
  opponent scores about 0.027 higher than the reverse), so it has mostly learned "who is stronger", the same thing the
  Elo baseline captures. Our default table uses equal ratings for both players, where the gap is zero.
* Likely reasons (not tested): ratings enter as just two numbers added at the very end of a 2,048-number board summary,
  only 5 epochs of training, and no clock or time-control information. Fixing this is a modelling job, not a difficulty-code job.

**So the question "do positions get easier as players get stronger?" cannot be answered with this checkpoint.**

## By game stage (`difficulty_by_stage.png`)

Opening (380 positions), middlegame (495) and endgame (125) all sit at about +0.04 and are flat across ratings.
Stage is defined by move number and non-pawn material (rules at the top of `explore_difficulty.py`).
No real difference between stages at this sample size.

## How spread out the scores are (`difficulty_spread.png`, rating 1600)

* Scores run from about -0.75 to +0.85. Middle 50% of positions: -0.17 to +0.28. Standard deviation 0.30.
* Only **19% of positions are within 0.05 of zero**, and 29% within 0.10. Slightly more than half (54%) are positive.
* The win-chance-only version is two humps (most positions near -0.45 or +0.35), because Stockfish says
  "win" or "loss" and the model never says either with much confidence.
* **Difficulty mostly tracks Stockfish's certainty.** 58% of positions are ones Stockfish already calls decided
  (score at most 0.05 or at least 0.95), and the correlation between difficulty and Stockfish's score is 0.89.
  In plain terms: most of what the score measures today is "the engine is sure, the model is not", not "humans find
  this specific position tricky".

## Hardest and easiest 25 positions at rating 1600 (`hardest_easiest_1600.html`, `.csv`)

* **Hardest**: Stockfish says the side to move is almost certainly winning (average 0.99), while the model gives only 0.34.
* **Easiest**: the reverse. Stockfish says the side to move is almost certainly losing (0.01), the model says 0.58.
  (A negative score is a sign the model and the engine disagree in the other direction, not that the position is "easy".)
* One pattern is statistically clear: **the side to move is in check in 24% of the hardest 25, versus 7% of all
  positions** (permutation test p = 0.006, with six comparisons made, so still treat it with care). Other features
  (piece count, material balance, stage, colour) were not clearly different from the overall mix; with 25 positions
  each, small differences there are noise.
* Hardest and easiest lists look alike in size and balance: about 23 to 25 pieces, material nearly level, and a mix
  of opening/middlegame/endgame. We did not find a simple rule for "hard".

## Limitations

1. **The model ignores rating** (above), so rating-based conclusions are not supported yet.
2. **Stockfish is an engine, not perfect play, and its win/draw/loss numbers are calibrated to engine games.** At
   200,000 nodes it is far stronger than any human; its probabilities are very decisive, which pushes many scores to the extremes.
   Forced mates (2.7% of positions) are just more extreme cases of this.
3. **The model learned from whole-game results**, not from "how hard was the next move". A position can look hard
   because the model learned that the side to move usually lost games from similar positions, for reasons that have
   nothing to do with this move.
4. **Training data we cannot see.** We do not know which data `model.pt` was trained on. Rating coverage was measured on our
   sample (trusted 1300 to 2000). Below 1200 there is no data at all.
5. **Missing inputs.** No clocks, no time control, no history, no move number in the model. Time pressure is the obvious
   thing that makes positions harder for humans, and the model cannot see it.
6. **Small and old sample.** 1,000 positions, about 250 games, one 2013 month, mixed time controls.
7. **Rebuilt positions are approximate in two ways:** the 50-move counter is reset to 0 and repetition history is lost.
8. **Not checked here:** whether difficulty matches thinking time, tablebases or puzzle ratings (Mary's validation).
