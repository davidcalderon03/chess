"""Evaluate the five losses in chess_five_task_loss_function.html."""
import argparse
import csv
import html
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LichessPositions
from model_io import load_checkpoint
from multitask_data import MultiTaskPositions
from multitask_model import DEFAULT_WEIGHTS, FiveTaskLoss
from multitask_training import evaluate_five_task
from train import pick_device

LOSSES = {
    "outcome": "Outcome CCE",
    "mistake": "Mistake BCE",
    "moves": "Remaining-length Huber",
    "time": "Thinking-time Huber",
    "termination": "Termination CCE",
}


@torch.inference_mode()
def evaluate_losses(model, loader, checkpoint, device):
    """Use the training loss; average each task over its valid held-out labels."""
    config = checkpoint.get("loss_config", {})
    weights = {**DEFAULT_WEIGHTS, **config.get("weights", {})}
    if checkpoint["inference_config"]["model_type"] == "five_task":
        criterion = FiveTaskLoss(checkpoint["normalization"], weights=weights,
                                 delta=config.get("huber_delta", 1.0))
        result = evaluate_five_task(model, loader, criterion, device)
        return {"losses": result["losses"], "valid_labels": result["counts"],
                "weights": weights, "total_loss": result["weighted_total"]}

    # Existing outcome-only checkpoints can still be evaluated.
    model.eval()
    loss_sum, count = 0.0, 0
    for board, context, labels in loader:
        logits = model(board.to(device), context.to(device))
        loss_sum += F.cross_entropy(logits, labels.to(device), reduction="sum").item()
        count += len(labels)
    losses = dict.fromkeys(LOSSES)
    counts = dict.fromkeys(LOSSES, 0)
    losses["outcome"], counts["outcome"] = loss_sum / count if count else None, count
    return {"losses": losses, "valid_labels": counts, "weights": weights, "total_loss": None}


def write_results(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = [{"task": task, "loss": name, "weight": result["weights"][task],
             "valid_labels": result["valid_labels"][task], "value": result["losses"][task]}
            for task, name in LOSSES.items()]
    rows.append({"task": "total", "loss": "Weighted total", "weight": None,
                 "valid_labels": None, "value": result["total_loss"]})
    with (directory / "metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def number(value):
        return "Unavailable" if value is None else f"{value:.6g}"

    cells = "".join(
        f"<tr><td>{row['loss']}</td><td>{number(row['weight']) if row['weight'] is not None else '—'}</td>"
        f"<td>{row['valid_labels'] if row['valid_labels'] is not None else '—'}</td>"
        f"<td>{number(row['value'])}</td></tr>" for row in rows)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Chess evaluation</title>
<style>body{{font:16px/1.5 system-ui;max-width:850px;margin:40px auto;padding:0 20px;color:#172b43}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #d6dfe7;text-align:left}}
nav{{margin:20px 0}}.table{{overflow-x:auto}}.run{{margin:18px 0;color:#52677a}}</style></head><body>
<h1>Evaluation losses</h1><div class="run">{html.escape(result['split'])} · {result['positions']:,} positions · {html.escape(Path(result['checkpoint']).name)}</div>
<nav><a href="metrics.json">JSON</a> · <a href="metrics.csv">CSV</a></nav>
<div class="table"><table><thead><tr><th>Loss</th><th>Weight</th><th>Valid labels</th><th>Value</th></tr></thead><tbody>{cells}</tbody></table></div>
</body></html>'''
    (directory / "report.html").write_text(page)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", help="shard directory; defaults to the checkpoint's data path")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", default="reports/evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    config = checkpoint["inference_config"]
    data = Path(args.data or checkpoint.get("args", {}).get("data", "data")).resolve()
    dataset_class = MultiTaskPositions if config["model_type"] == "five_task" else LichessPositions
    dataset = dataset_class(data, args.split, features=config["features"], history=config["history"])
    if not len(dataset):
        raise ValueError("The selected partition has no positions")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    result = evaluate_losses(model, loader, checkpoint, device)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), data=str(data),
                  split=args.split, positions=len(dataset))
    write_results(result, args.out)
    if not args.quiet:
        for task, name in LOSSES.items():
            value = result["losses"][task]
            print(f"{name:<24} {'Unavailable' if value is None else f'{value:.6f}'}")
        total = result["total_loss"]
        print(f"{'Weighted total':<24} {'Unavailable' if total is None else f'{total:.6f}'}")
        print(f"Saved: {Path(args.out) / 'report.html'}")
    return result


if __name__ == "__main__":
    run(parse_args())
