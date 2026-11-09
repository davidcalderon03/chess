# Rating sweep (checkpoint model.pt, 20 test positions)

Side to move's model expected score (win + 0.5 draw) at ratings [1200, 1400, 1600, 1800, 2000, 2200, 2400]. Ratings outside 1300-2000 are outside what the (sample) training data covers well.

| group | n | mean score at lowest rating | at highest rating | change | positions moving the 'right' way |
|---|---|---|---|---|---|
| winning | 8 | 0.7106 | 0.7111 | +0.0005 | 5/8 |
| losing | 8 | 0.2291 | 0.2315 | +0.0024 | 4/8 |
| balanced | 4 | 0.5300 | 0.5317 | +0.0017 | 4/4 |

## Per position (score at each rating; Stockfish score for reference)

| pos_id | group | SF score | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 |
|---|---|---|---|---|---|---|---|---|---|
| test:shard_00000.npz:25610 | winning | 1.00 | 0.746 | 0.747 | 0.747 | 0.747 | 0.748 | 0.748 | 0.749 |
| test:shard_00000.npz:2818 | winning | 1.00 | 0.888 | 0.888 | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 |
| test:shard_00000.npz:1026 | winning | 1.00 | 0.631 | 0.632 | 0.632 | 0.633 | 0.633 | 0.633 | 0.634 |
| test:shard_00000.npz:17111 | winning | 1.00 | 0.576 | 0.576 | 0.575 | 0.575 | 0.575 | 0.575 | 0.574 |
| test:shard_00000.npz:10305 | winning | 1.00 | 0.969 | 0.969 | 0.969 | 0.969 | 0.969 | 0.969 | 0.969 |
| test:shard_00000.npz:9251 | winning | 1.00 | 0.746 | 0.746 | 0.746 | 0.746 | 0.746 | 0.745 | 0.745 |
| test:shard_00000.npz:1867 | winning | 1.00 | 0.600 | 0.601 | 0.601 | 0.601 | 0.601 | 0.601 | 0.601 |
| test:shard_00000.npz:20114 | winning | 0.96 | 0.528 | 0.528 | 0.528 | 0.528 | 0.528 | 0.528 | 0.528 |
| test:shard_00000.npz:17471 | losing | 0.00 | 0.202 | 0.203 | 0.204 | 0.205 | 0.206 | 0.207 | 0.207 |
| test:shard_00000.npz:19288 | losing | 0.00 | 0.149 | 0.149 | 0.149 | 0.149 | 0.150 | 0.150 | 0.150 |
| test:shard_00000.npz:28827 | losing | 0.00 | 0.085 | 0.085 | 0.084 | 0.084 | 0.084 | 0.084 | 0.084 |
| test:shard_00000.npz:20686 | losing | 0.00 | 0.327 | 0.327 | 0.328 | 0.328 | 0.328 | 0.328 | 0.328 |
| test:shard_00000.npz:25075 | losing | 0.00 | 0.367 | 0.367 | 0.367 | 0.367 | 0.367 | 0.367 | 0.367 |
| test:shard_00000.npz:9178 | losing | 0.02 | 0.472 | 0.472 | 0.471 | 0.471 | 0.471 | 0.470 | 0.470 |
| test:shard_00000.npz:17153 | losing | 0.00 | 0.173 | 0.177 | 0.180 | 0.182 | 0.185 | 0.187 | 0.190 |
| test:shard_00000.npz:22127 | losing | 0.00 | 0.057 | 0.057 | 0.057 | 0.056 | 0.056 | 0.056 | 0.056 |
| test:shard_00000.npz:3123 | balanced | 0.47 | 0.484 | 0.484 | 0.484 | 0.484 | 0.484 | 0.484 | 0.484 |
| test:shard_00000.npz:27357 | balanced | 0.51 | 0.539 | 0.539 | 0.539 | 0.540 | 0.540 | 0.540 | 0.540 |
| test:shard_00000.npz:5933 | balanced | 0.52 | 0.550 | 0.551 | 0.551 | 0.552 | 0.553 | 0.553 | 0.554 |
| test:shard_00000.npz:26627 | balanced | 0.58 | 0.547 | 0.547 | 0.547 | 0.548 | 0.548 | 0.548 | 0.548 |

Largest change in any single position's score across the whole sweep: 0.0165; median 0.0014.

## Verdict

The model does NOT respond to rating in a meaningful way: predicted results barely move from 1200 to 2400, so it does not show stronger players converting winning positions more often.
