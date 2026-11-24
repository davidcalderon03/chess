"""
Informative baselines for the five-task model, scored with the same losses as evaluate.py.

    python bot_baselines.py --data data_five --model-metrics reports/five_test/metrics.json
    python bot_baselines.py --data data_five --model-metrics reports/five_test/metrics.json --maia2 rapid

Every baseline is fitted on the train split and scored on --split (default test). Rows:

    No information          intercept only: class frequencies / mean target (the old baseline)
    Elo only                both players' ratings
    Hand-built features     ratings, clocks, time control, move number, material, side to move
    Stockfish (calibrated)  Stockfish's expected score for the position, stored by
                            `extract.py --tasks all` (no new engine runs); outcome and mistake only
    Stockfish + hand-built  Stockfish's expected score plus the hand-built features; all tasks
    Maia-2 (calibrated)     with --maia2: Maia-2's value and move-policy confidence for the
                            players' ratings; outcome and mistake only
    Model                   with --model-metrics: the evaluate.py results to compare against

"Calibrated" means a small (multinomial) logistic regression maps the bot's raw output to the
task's labels, fitted on the train split. Without it Stockfish would be scored on engine-play
probabilities (far too confident about human games), which is unfair to it.

Categorical tasks use (multinomial) logistic regression, the regression tasks a linear model on
the same standardized log1p target as training. Outputs go to --out: metrics.json, metrics.csv
and table.tex (a LaTeX tabular for the report).

Maia-2 (pip package `maia2`) pins torch < 2.9, so install it without letting pip change torch:
    pip install --no-deps maia2 && pip install einops gdown pandas pyyaml pyzstd tqdm
It downloads its weights on first use into --maia2-dir.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation_metrics import (DEFAULT_WEIGHTS, binary_metrics, categorical_metrics,
                                five_task_loss, regression_metrics)

TASKS = ("outcome", "mistake", "moves", "time", "termination")
LOSS_NAMES = {"outcome": "Outcome CCE", "mistake": "Mistake BCE", "moves": "Remaining-length Huber",
              "time": "Thinking-time Huber", "termination": "Termination CCE"}
PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], float)  # P N B R Q K
KEYS = ("bb", "meta", "elo", "clk", "clk_mask", "tc", "ply", "fen", "y", "mistake_y", "mistake_mask",
        "mistake_delta", "mistake_best_expected_score", "mistake_human_expected_score",
        "remaining_plies", "next_move_seconds", "time_mask", "termination_y", "termination_mask")


# ---------------------------------------------------------------- data

def load_split(root, split):
    paths = sorted(Path(root, split).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards in {root}/{split}")
    parts = [np.load(p, allow_pickle=False) for p in paths]
    missing = set(KEYS) - set(parts[0].files)
    if missing:
        raise ValueError(f"{root}/{split} lacks {sorted(missing)}; extract it with --tasks all")
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def labels_and_masks(d):
    return ({"outcome": d["y"].astype(int), "mistake": d["mistake_y"].astype(int),
             "moves": d["remaining_plies"].astype(float), "time": d["next_move_seconds"].astype(float),
             "termination": d["termination_y"].astype(int)},
            {"outcome": np.ones(len(d["y"]), bool), "mistake": d["mistake_mask"].astype(bool),
             "moves": np.ones(len(d["y"]), bool), "time": d["time_mask"].astype(bool),
             "termination": d["termination_mask"].astype(bool)})


def normalization(train):
    """Training-only log1p mean/std, as multitask_data.training_normalization computes it."""
    y, m = labels_and_masks(train)
    out = {}
    for task in ("moves", "time"):
        logged = np.log1p(y[task][m[task]])
        std = float(logged.std())
        out[task] = {"mean": float(logged.mean()), "std": std if std > 1e-8 else 1.0}
    return out


# ---------------------------------------------------------------- features

def stockfish_value(d):
    """Stockfish expected score (W + D/2) for the side to move, and whether it is available.
    extract.py stores it in mistake_best_expected_score; a failed search leaves all zeros."""
    s = d["mistake_best_expected_score"].astype(float)
    failed = ~d["mistake_mask"] & (s == 0) & (d["mistake_human_expected_score"] == 0) & (d["mistake_delta"] == 0)
    return np.where(failed, 0.5, s), (~failed).astype(float)


def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def hand_features(d):
    """White-perspective features for outcome/termination, mover-perspective ones for the
    move-level tasks (mistake, think time). Returns (white_view, mover_view) matrices."""
    white_to_move = d["meta"][:, 0] == 1
    side = np.where(white_to_move, 1.0, -1.0)
    counts = np.unpackbits(np.ascontiguousarray(d["bb"]).view(np.uint8).reshape(len(side), 12, 8),
                           axis=2).sum(2).astype(float)
    material_w, material_b = counts[:, :6] @ PIECE_VALUES, counts[:, 6:] @ PIECE_VALUES
    elo = d["elo"].astype(float)
    clk_ok = d["clk_mask"].astype(bool)
    clk = np.where(clk_ok, np.log1p(np.maximum(d["clk"].astype(float), 0)), 0.0)
    base, inc = np.log1p(d["tc"][:, 0].astype(float)), d["tc"][:, 1].astype(float) / 60
    ply = d["ply"].astype(float) / 100
    white = np.column_stack([
        (elo[:, 0] - elo[:, 1]) / 400, (elo.mean(1) - 1500) / 500, side,
        (material_w - material_b) / 10, (material_w + material_b) / 78, ply,
        clk[:, 0] - clk[:, 1], clk[:, 0], clk[:, 1], clk_ok.all(1).astype(float), base, inc])
    me, opp = np.where(white_to_move, 0, 1), np.where(white_to_move, 1, 0)
    rows = np.arange(len(side))
    mover = np.column_stack([
        (elo[rows, me] - 1500) / 500, (elo[rows, opp] - 1500) / 500,
        side * (material_w - material_b) / 10, (material_w + material_b) / 78, ply,
        clk[rows, me], clk[rows, opp], clk[rows, me] - base, clk_ok[rows, me].astype(float), base, inc])
    return white, mover


def elo_features(d):
    elo = d["elo"].astype(float)
    white_to_move = d["meta"][:, 0] == 1
    mover = np.where(white_to_move, elo[:, 0], elo[:, 1])
    other = np.where(white_to_move, elo[:, 1], elo[:, 0])
    return (np.column_stack([(elo[:, 0] - elo[:, 1]) / 400, (elo.mean(1) - 1500) / 500]),
            np.column_stack([(mover - 1500) / 500, (other - 1500) / 500]))


def stockfish_features(d):
    s, ok = stockfish_value(d)
    white_to_move = d["meta"][:, 0] == 1
    s_white = np.where(white_to_move, s, 1 - s)
    white = np.column_stack([logit(s_white), ok])
    mover = np.column_stack([logit(s), np.abs(s - 0.5), ok])  # decisive positions make mistakes cheaper/costlier
    return white, mover


def maia2_features(d, model_type, save_dir, device, batch=512):
    """Maia-2 value (White's expected score) and policy confidence at the players' ratings."""
    from maia2 import inference, model as maia_model

    net = maia_model.from_pretrained(type=model_type, device="gpu" if device.type == "cuda" else "cpu",
                                     save_root=str(save_dir))
    net.eval()
    all_moves_dict, elo_dict, _ = inference.prepare()
    dev = next(net.parameters()).device
    fens = [str(f) for f in d["fen"]]
    elo = d["elo"].astype(int)
    white_to_move = d["meta"][:, 0] == 1
    value, top, entropy = np.zeros(len(fens)), np.zeros(len(fens)), np.zeros(len(fens))
    with torch.no_grad():
        for start in range(0, len(fens), batch):
            idx = range(start, min(start + batch, len(fens)))
            items = [inference.preprocessing(fens[i], elo[i, 0 if white_to_move[i] else 1],
                                             elo[i, 1 if white_to_move[i] else 0], elo_dict, all_moves_dict)
                     for i in idx]
            boards, me, opp, legal = (torch.stack([it[0] for it in items]).to(dev),
                                      torch.tensor([it[1] for it in items]).to(dev),
                                      torch.tensor([it[2] for it in items]).to(dev),
                                      torch.stack([it[3] for it in items]).to(dev).bool())
            logits_policy, _, logits_value = net(boards, me, opp)
            p = logits_policy.masked_fill(~legal, float("-inf")).softmax(-1)
            v = (logits_value / 2 + 0.5).clamp(0, 1).flatten().cpu().numpy()  # side to move
            sl = slice(idx.start, idx.stop)
            value[sl] = v
            top[sl] = p.max(-1).values.cpu().numpy()
            entropy[sl] = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).cpu().numpy()
            if (start // batch) % 20 == 0:
                print(f"  maia2 {idx.stop:,}/{len(fens):,}", flush=True)
    v_white = np.where(white_to_move, value, 1 - value)
    return (np.column_stack([logit(v_white)]),
            np.column_stack([logit(value), np.abs(value - 0.5), top, entropy]))


# ---------------------------------------------------------------- models

class Standardizer:
    def __init__(self, x):
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[self.std < 1e-8] = 1.0

    def __call__(self, x):
        return (x - self.mean) / self.std


def fit_softmax(x, y, classes, l2=1e-4):
    """Multinomial logistic regression (bias + weights) by L-BFGS; returns a predict_proba function."""
    scale = Standardizer(x) if x.shape[1] else None
    X = torch.tensor(scale(x) if scale else x, dtype=torch.float64)
    Y = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, Y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    W, b = W.detach(), b.detach()

    def predict(x_new):
        Xn = torch.tensor(scale(x_new) if scale else x_new, dtype=torch.float64)
        return torch.softmax(Xn @ W + b, 1).numpy()
    return predict


def fit_linear(x, z, l2=1e-4):
    """Ridge regression on the standardized target; returns a predict function (standardized units)."""
    scale = Standardizer(x) if x.shape[1] else None
    X = np.column_stack([np.ones(len(z)), scale(x) if scale else x])
    reg = l2 * len(z) * np.eye(X.shape[1])
    reg[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ z)
    return lambda x_new: np.column_stack([np.ones(len(x_new)), scale(x_new) if scale else x_new]) @ beta


def score_task(task, x_train, x_eval, train, evals, norm):
    """Fit one baseline for one task on train and return its evaluation loss (and the metric dict)."""
    y_tr, m_tr = labels_and_masks(train)
    y_ev, m_ev = labels_and_masks(evals)
    if task in ("outcome", "termination"):
        predict = fit_softmax(x_train[m_tr[task]], y_tr[task][m_tr[task]], 3)
        p = predict(x_eval)
        names = ["white_win", "draw", "black_win"] if task == "outcome" else ["checkmate", "time_forfeit", "other_normal"]
        metric = categorical_metrics(p, np.where(m_ev[task], y_ev[task], 0), names, mask=m_ev[task])
        return metric["categorical_cross_entropy"], metric
    if task == "mistake":
        predict = fit_softmax(x_train[m_tr[task]], y_tr[task][m_tr[task]], 2)
        q = predict(x_eval)[:, 1]
        metric = binary_metrics(q, np.where(m_ev[task], y_ev[task], 0), mask=m_ev[task])
        return metric["binary_cross_entropy"], metric
    mean, std = norm[task]["mean"], norm[task]["std"]
    z = (np.log1p(y_tr[task][m_tr[task]]) - mean) / std
    predict = fit_linear(x_train[m_tr[task]], z)
    metric = regression_metrics(predict(x_eval), np.where(m_ev[task], y_ev[task], 0), mean=mean, std=std,
                                predictions_standardized=True, mask=m_ev[task])
    return metric["huber_loss"], metric


# ---------------------------------------------------------------- report

def write_outputs(rows, counts, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"rows": rows, "valid_labels": counts}, indent=2) + "\n")
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", *TASKS, "weighted_total"])
        for r in rows:
            w.writerow([r["name"], *[r["losses"].get(t) for t in TASKS], r["total"]])

    best = {t: min((r["losses"][t] for r in rows if r["losses"].get(t) is not None), default=None) for t in TASKS}

    def cell(r, t):
        v = r["losses"].get(t)
        if v is None:
            return "---"
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if best[t] is not None and s == f"{best[t]:.3f}" else s

    lines = [r"\begin{tabular}{lrrrrrr}", r"  \toprule",
             r"  Model & Outcome & Mistake & Length & Time & Termination & Weighted \\",
             r"  \midrule"]
    for r in rows:
        if r["name"] == "Model":
            lines.append(r"  \midrule")
        total = "---" if r["total"] is None else f"{r['total']:.3f}"
        lines.append(f"  {r['label']} & " + " & ".join(cell(r, t) for t in TASKS) + f" & {total} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    (out / "table.tex").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out / 'metrics.json'}, {out / 'metrics.csv'}, {out / 'table.tex'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data_five")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--model-metrics", help="evaluate.py metrics.json to add as the 'Model' row")
    p.add_argument("--maia2", choices=("rapid", "blitz"), help="add a Maia-2 row (needs the maia2 package)")
    p.add_argument("--maia2-dir", default="maia2_models", help="where Maia-2 weights are downloaded")
    p.add_argument("--maia2-train", type=int, default=20_000,
                   help="train positions used to calibrate Maia-2 (0 = all)")
    p.add_argument("--out", default="reports/baselines")
    a = p.parse_args()

    train, evals = load_split(a.data, "train"), load_split(a.data, a.split)
    norm = normalization(train)
    _, m_ev = labels_and_masks(evals)
    counts = {t: int(m_ev[t].sum()) for t in TASKS}
    print(f"fit on {len(train['y']):,} train positions, scored on {len(evals['y']):,} {a.split} positions")

    def view(task, white, mover):
        return mover if task in ("mistake", "time") else white

    feats = {}
    for name, fn in (("hand", hand_features), ("elo", elo_features), ("sf", stockfish_features)):
        feats[name] = (fn(train), fn(evals))
    sf_ok = stockfish_value(evals)[1]
    print(f"Stockfish value available for {sf_ok.mean():.1%} of {a.split} positions")

    def combine(*names):
        return tuple(tuple(np.column_stack([feats[n][s][v] for n in names]) for v in range(2)) for s in range(2))

    empty = tuple((np.zeros((len(d["y"]), 0)),) * 2 for d in (train, evals))
    specs = [("No information", "No information", empty, TASKS),
             ("Elo only", "Elo only", feats["elo"], TASKS),
             ("Hand-built features", "Hand-built features", feats["hand"], TASKS),
             ("Stockfish (calibrated)", "Stockfish, calibrated", feats["sf"], ("outcome", "mistake")),
             ("Stockfish + hand-built", "Stockfish + hand-built", combine("sf", "hand"), TASKS)]

    if a.maia2:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sub = np.arange(len(train["y"]))
        if a.maia2_train and a.maia2_train < len(sub):
            sub = np.sort(np.random.default_rng(0).choice(len(sub), a.maia2_train, replace=False))
        train_sub = {k: v[sub] for k, v in train.items()}
        print(f"Maia-2 ({a.maia2}) on {len(sub):,} train + {len(evals['y']):,} {a.split} positions ...")
        maia = (maia2_features(train_sub, a.maia2, a.maia2_dir, device),
                maia2_features(evals, a.maia2, a.maia2_dir, device))
        specs.append((f"Maia-2 {a.maia2} (calibrated)", f"Maia-2 ({a.maia2}), calibrated", maia,
                      ("outcome", "mistake"), train_sub))

    rows = []
    for spec in specs:
        name, label, (xtr, xev), tasks = spec[:4]
        fit_on = spec[4] if len(spec) > 4 else train
        losses = {}
        for task in TASKS:
            if task in tasks:
                losses[task], _ = score_task(task, view(task, *xtr), view(task, *xev), fit_on, evals, norm)
            else:
                losses[task] = None
        total = five_task_loss({t: {"n": counts[t], **({k: losses[t]} if losses[t] is not None else {})}
                                for t, k in zip(TASKS, ("categorical_cross_entropy", "binary_cross_entropy",
                                                        "huber_loss", "huber_loss",
                                                        "categorical_cross_entropy"))})["total"]
        rows.append({"name": name, "label": label, "losses": losses, "total": total})

    if a.model_metrics:
        m = json.loads(Path(a.model_metrics).read_text())
        if m.get("split") not in (None, a.split):
            print(f"warning: {a.model_metrics} is for the {m['split']} split, not {a.split}")
        rows.append({"name": "Model", "label": "Our model", "losses": {t: m["losses"].get(t) for t in TASKS},
                     "total": m.get("total_loss")})

    print(f"\n{'':<26}" + "".join(f"{LOSS_NAMES[t]:>24}" for t in TASKS) + f"{'Weighted total':>16}")
    print(f"{'valid labels':<26}" + "".join(f"{counts[t]:>24,}" for t in TASKS))
    def fmt(v):
        return "---" if v is None else f"{v:.4f}"

    for r in rows:
        print(f"{r['name']:<26}" + "".join(f"{fmt(r['losses'][t]):>24}" for t in TASKS)
              + f"{fmt(r['total']):>16}")
    print(f"(weights {DEFAULT_WEIGHTS}; lower is better; '---' = baseline does not cover that task)")
    write_outputs(rows, counts, a.out)


if __name__ == "__main__":
    main()
