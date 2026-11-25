This file is kept intentionally to show the results after some variations on the original approach.

## Default
```
 python train.py --data data_five --tasks all --features board+skill+time --epochs 10 \
    --save checkpoints/Nov25.pt --no-show --skip-test && rm checkpoints/Nov25.pt.history.json&& rm checkpoints/Nov25.pt
Device: cuda | Parameters: 2,326,601 | Inputs: board+skill+time
Positions: train 90,021, validation 5,005 | 176 batches/epoch
Epoch   Train total   Val total   Train CCE   Val CCE   Val acc   Time
  1/10       1.3635      1.3450      0.8326    0.8128    52.1%    4.1s  best
  2/10       1.3255      1.3190      0.8075    0.7936    57.0%    3.7s  best
  3/10       1.3085      1.3099      0.7957    0.7902    56.7%    3.8s  best
  4/10       1.2917      1.3017      0.7839    0.7856    57.0%    3.7s  best
  5/10       1.2737      1.2842      0.7705    0.7724    58.7%    3.7s  best
  6/10       1.2566      1.2769      0.7578    0.7676    58.5%    3.8s  best
  7/10       1.2356      1.3439      0.7419    0.8262    56.1%    3.7s
  8/10       1.2063      1.2831      0.7179    0.7794    59.0%    3.7s
  9/10       1.1584      1.2994      0.6762    0.7957    59.5%    3.7s
 10/10       1.1107      1.3216      0.6328    0.8172    58.6%    3.7s
Finished 10 epochs, 1,760 batches. Best: epoch 6 (validation outcome CCE 0.7676).
Checkpoint: checkpoints/Nov25.pt
History: checkpoints/Nov25.pt.history.json
Curves: loss_curve.png
```

## Default + Data Augmentation (with --augment flag)
```
 python train.py --data data_five --tasks all --features board+skill+time --epochs 10 \
    --save checkpoints/Nov25.pt --no-show --skip-test --augment && rm checkpoints/Nov25.pt.history.json&& rm checkpoints/Nov25.pt
Device: cuda | Parameters: 2,326,601 | Inputs: board+skill+time
Positions: train 90,021, validation 5,005 | 176 batches/epoch
Epoch   Train total   Val total   Train CCE   Val CCE   Val acc   Time
  1/10       1.3634      1.3455      0.8321    0.8067    53.1%    4.2s  best
  2/10       1.3262      1.3215      0.8083    0.7992    56.4%    3.7s  best
  3/10       1.3098      1.3065      0.7968    0.7819    57.9%    3.8s  best
  4/10       1.2943      1.3104      0.7856    0.7885    56.3%    3.8s
  5/10       1.2827      1.3220      0.7785    0.7946    56.5%    3.7s
  6/10       1.2673      1.2677      0.7686    0.7616    60.0%    3.8s  best
  7/10       1.2558      1.2627      0.7612    0.7594    58.8%    3.7s  best
  8/10       1.2424      1.2540      0.7526    0.7542    59.7%    3.8s  best
  9/10       1.2293      1.2478      0.7432    0.7516    60.0%    3.7s  best
 10/10       1.2199      1.2443      0.7364    0.7497    60.6%    3.8s  best
Finished 10 epochs, 1,760 batches. Best: epoch 10 (validation outcome CCE 0.7497).
Checkpoint: checkpoints/Nov25.pt
History: checkpoints/Nov25.pt.history.json
Curves: loss_curve.png
```

## Default + Data Augmentation + 3rd layer
```
 python train.py --data data_five --tasks all --features board+skill+time --epochs 10 \
    --save checkpoints/Nov25.pt --no-show --skip-test --augment && rm checkpoints/Nov25.pt.history.json&& rm checkpoints/Nov25.pt
Device: cuda | Parameters: 3,212,873 | Inputs: board+skill+time
Positions: train 90,021, validation 5,005 | 176 batches/epoch
Epoch   Train total   Val total   Train CCE   Val CCE   Val acc   Time
  1/10       1.3585      1.3367      0.8294    0.8040    53.2%    5.5s  best
  2/10       1.3278      1.3302      0.8095    0.8045    55.6%    5.0s
  3/10       1.3109      1.3105      0.7977    0.7885    54.0%    5.1s  best
  4/10       1.2951      1.2858      0.7865    0.7711    59.2%    5.1s  best
  5/10       1.2823      1.2982      0.7783    0.7826    55.7%    5.1s
  6/10       1.2664      1.2628      0.7682    0.7559    60.7%    5.2s  best
  7/10       1.2560      1.2531      0.7618    0.7518    61.3%    5.1s  best
  8/10       1.2420      1.2458      0.7525    0.7488    60.8%    5.1s  best
  9/10       1.2295      1.2415      0.7440    0.7479    60.8%    5.1s  best
 10/10       1.2222      1.2374      0.7389    0.7451    61.9%    5.1s  best
Finished 10 epochs, 1,760 batches. Best: epoch 10 (validation outcome CCE 0.7451).
Checkpoint: checkpoints/Nov25.pt
History: checkpoints/Nov25.pt.history.json
Curves: loss_curve.png
```


## Default + Data Augmentation + 3rd layer + Changing the Adam Weight Decay
Same results as above

## Default + Data Augmentation + 3rd layer + SGD Instead of Adam
Same results as above