"""NumPy evaluation for the proposed five-task objective.

Class order is supplied explicitly; current outcome order is White/draw/Black.
Masks are explicit: no function silently discards a NaN or reshapes an input.
Cross-entropy uses natural logs and clips zero event probabilities at 1e-15.
"""
import numpy as np


DEFAULT_WEIGHTS = {"outcome": 1.0, "mistake": 0.5, "moves": 0.25,
                   "time": 0.25, "termination": 0.1}
LOSS_KEYS = {"outcome": "categorical_cross_entropy", "mistake": "binary_cross_entropy",
             "moves": "huber_loss", "time": "huber_loss", "termination": "categorical_cross_entropy"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _selection(n, mask):
    selected = np.ones(n, dtype=bool) if mask is None else np.asarray(mask)
    _require(selected.shape == (n,), "mask must have shape (N,)")
    _require(np.isin(selected, [False, True]).all(), "mask must contain only boolean/0/1 values")
    return selected.astype(bool)


def _coverage(n, selected):
    valid = int(selected.sum())
    return {"n": valid, "total_count": n, "masked_count": n - valid}


def _labels(values, n, classes, selected):
    values = np.asarray(values)
    _require(values.shape == (n,), "labels must have shape (N,)")
    values = np.asarray(values[selected], dtype=np.float64)
    _require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(),
             "selected labels must be finite integer class indices")
    _require(((values >= 0) & (values < classes)).all(), "selected class label is out of range")
    return values.astype(np.int64)


def _probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] >= 2, "probabilities must have shape (N, C), C>=2")
    _require(np.isfinite(values).all(), "probabilities must be finite")
    _require(((values >= 0) & (values <= 1)).all(), "probabilities must lie in [0,1]")
    _require(np.allclose(values.sum(1), 1, rtol=0, atol=1e-6), "probability rows must sum to 1")
    return values


def _reliability(prediction, observation, bins):
    _require(isinstance(bins, (int, np.integer)) and bins > 0, "bins must be a positive integer")
    index = np.minimum((prediction * bins).astype(int), bins - 1)
    rows, error = [], 0.0
    for b in range(bins):
        use = index == b
        n = int(use.sum())
        predicted = float(prediction[use].mean()) if n else None
        observed = float(observation[use].mean()) if n else None
        rows.append({"lower": b / bins, "upper": (b + 1) / bins, "count": n,
                     "predicted_mean": predicted, "observed_mean": observed})
        if n:
            error += n / len(prediction) * abs(predicted - observed)
    return {"ece": float(error) if len(prediction) else None, "bins": rows}


def categorical_metrics(probabilities, labels, class_names=None, bins=10, *, mask=None):
    p = _probabilities(probabilities)
    total, classes = p.shape
    names = list(class_names) if class_names is not None else [str(i) for i in range(classes)]
    _require(len(names) == classes and len(set(names)) == classes
             and all(isinstance(name, str) for name in names), "class_names must be unique strings matching C")
    use = _selection(total, mask)
    y, p = _labels(labels, total, classes, use), p[use]
    predicted = p.argmax(1)
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    per_class, reliability = {}, {}
    for k, name in enumerate(names):
        tp, support, predicted_count = int(matrix[k, k]), int(matrix[k].sum()), int(matrix[:, k].sum())
        per_class[name] = {"precision": tp / predicted_count if predicted_count else None,
                           "recall": tp / support if support else None,
                           "f1": 2 * tp / (support + predicted_count) if support + predicted_count else 0.0,
                           "support": support}
        reliability[name] = _reliability(p[:, k], y == k, bins)
    recalls = [row["recall"] for row in per_class.values() if row["support"]]
    result = {**_coverage(total, use), "class_names": names,
              "categorical_cross_entropy": None, "accuracy": None, "brier_score": None,
              "macro_f1": None, "balanced_accuracy": None, "per_class": per_class,
              "confusion_matrix": matrix.tolist(), "reliability_by_class": reliability,
              "top_label_ece": _reliability(p.max(1), predicted == y, bins)["ece"],
              "macro_f1_policy": "all declared classes; F1=0 for an absent/unpredicted class",
              "balanced_accuracy_policy": "mean recall among classes with true support"}
    if len(y):
        result.update(categorical_cross_entropy=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                      accuracy=float((predicted == y).mean()),
                      brier_score=float(((p - np.eye(classes)[y]) ** 2).sum(1).mean()),
                      macro_f1=float(np.mean([row["f1"] for row in per_class.values()])),
                      balanced_accuracy=float(np.mean(recalls)))
    return result


def _ranking(q, y):
    positives, negatives = int(y.sum()), len(y) - int(y.sum())
    if not positives:
        return None, None
    order = np.argsort(-q, kind="stable")
    scores, labels = q[order], y[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores)) + 1, len(y)]
    tp = fp = 0
    ap = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        added = int(labels[start:end].sum())
        tp += added
        fp += end - start - added
        ap += added / positives * tp / (tp + fp)
    if not negatives:
        return float(ap), None
    # A positive outranks all negatives in lower-score groups, and half its ties.
    lower_negatives, concordant = negatives, 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        positive = int(labels[start:end].sum())
        negative = end - start - positive
        lower_negatives -= negative
        concordant += positive * (lower_negatives + 0.5 * negative)
    return float(ap), float(concordant / (positives * negatives))


def binary_metrics(probabilities, labels, threshold=0.5, bins=10, *, mask=None):
    q = np.asarray(probabilities, dtype=np.float64)
    _require(q.ndim == 1, "binary probabilities must have shape (N,)")
    _require(np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all(), "binary probabilities must be finite in [0,1]")
    _require(np.isfinite(threshold) and 0 <= threshold <= 1, "threshold must lie in [0,1]")
    use = _selection(len(q), mask)
    y, q = _labels(labels, len(q), 2, use), q[use]
    positive = q >= threshold
    tp, fp = int(((y == 1) & positive).sum()), int(((y == 0) & positive).sum())
    fn, tn = int(((y == 1) & ~positive).sum()), int(((y == 0) & ~positive).sum())
    ap, auc = _ranking(q, y) if len(y) else (None, None)
    return {**_coverage(len(use), use), "threshold": float(threshold),
            "binary_cross_entropy": float(-np.log(np.clip(np.where(y == 1, q, 1 - q), 1e-15, 1)).mean()) if len(y) else None,
            "brier_score": float(((q - y) ** 2).mean()) if len(y) else None,
            "average_precision": ap, "auroc": auc,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "positive_prevalence": float(y.mean()) if len(y) else None,
            "positive_count": tp + fn, "negative_count": tn + fp,
            "reliability": _reliability(q, y, bins)}


def regression_metrics(predictions, targets, *, mean=None, std=None, delta=1.0,
                       predictions_standardized=False, mask=None):
    predicted, actual = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    _require(predicted.ndim == 1 and actual.shape == predicted.shape, "regression arrays must both have shape (N,)")
    _require(np.isfinite(delta) and delta > 0, "Huber delta must be finite and positive")
    normalized = mean is not None or std is not None
    if normalized:
        _require(mean is not None and std is not None and np.isfinite(mean)
                 and np.isfinite(std) and std > 0, "mean and positive std must both be finite training statistics")
    _require(not predictions_standardized or normalized, "standardized predictions require mean and std")
    use = _selection(len(actual), mask)
    predicted, actual = predicted[use], actual[use]
    _require(np.isfinite(predicted).all() and np.isfinite(actual).all(), "selected regression values must be finite")
    _require((actual >= 0).all(), "original-unit targets must be nonnegative")
    z_predicted = predicted.copy() if predictions_standardized else None
    if predictions_standardized:
        try:
            with np.errstate(over="raise", invalid="raise"):
                predicted = np.maximum(0, np.expm1(mean + std * predicted))
        except FloatingPointError as error:
            raise ValueError("inverse log transform overflowed") from error
        _require(np.isfinite(predicted).all(), "inverse log transform must be finite")
    else:
        _require((predicted >= 0).all(), "original-unit predictions must be nonnegative")
    result = {**_coverage(len(use), use), "mean_absolute_error": None,
              "root_mean_squared_error": None, "huber_loss": None, "huber_delta": float(delta)}
    if len(actual):
        errors = predicted - actual
        _require(np.isfinite(errors ** 2).all(), "regression errors overflowed")
        result.update(mean_absolute_error=float(np.abs(errors).mean()),
                      root_mean_squared_error=float(np.sqrt((errors ** 2).mean())))
        if normalized:
            if z_predicted is None:
                z_predicted = (np.log1p(predicted) - mean) / std
            residual = np.abs(z_predicted - (np.log1p(actual) - mean) / std)
            huber = np.where(residual <= delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
            _require(np.isfinite(huber).all(), "standardized Huber loss must be finite")
            result["huber_loss"] = float(huber.mean())
    return result


def five_task_loss(task_metrics, weights=None):
    coefficients = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        _require(not (set(weights) - set(coefficients)), "unknown task weight")
        coefficients.update(weights)
    _require(all(np.isfinite(value) and value >= 0 for value in coefficients.values()),
             "task weights must be finite and nonnegative")
    losses, available = {}, {}
    for task, key in LOSS_KEYS.items():
        metric = task_metrics.get(task) or {}
        value = metric.get(key)
        ok = metric.get("n", 0) > 0 and value is not None and np.isfinite(value) and value >= 0
        available[task], losses[task] = bool(ok), float(value) if ok else None
    missing = [task for task in coefficients if not available[task]]
    total = None if missing else float(sum(coefficients[task] * losses[task] for task in coefficients))
    return {"total": total, "weights": coefficients, "available": available,
            "missing": missing, "task_losses": losses}


def softmax(logits, T=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    _require(logits.ndim == 2 and logits.shape[1] >= 2 and len(logits) > 0,
             "logits must have nonempty shape (N,C), C>=2")
    _require(np.isfinite(logits).all() and np.isfinite(T) and T > 0, "finite logits and positive temperature required")
    weights = np.exp((logits - logits.max(1, keepdims=True)) / T)
    return weights / weights.sum(1, keepdims=True)


def categorical_cross_entropy_from_logits(logits, labels, *, mask=None):
    """Stable task mean without the reporting probability floor; None if empty."""
    logits = np.asarray(logits, dtype=np.float64)
    _require(logits.ndim == 2 and logits.shape[1] >= 2, "logits must have shape (N,C), C>=2")
    _require(np.isfinite(logits).all(), "logits must be finite")
    use = _selection(len(logits), mask)
    y, logits = _labels(labels, len(logits), logits.shape[1], use), logits[use]
    if not len(y):
        return None
    shifted = logits - logits.max(1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]
    _require(np.isfinite(losses).all(), "categorical cross-entropy overflowed")
    return float(losses.mean())


def binary_cross_entropy_from_logits(logits, labels, *, mask=None):
    """Stable BCEWithLogits task mean, including extreme confident mistakes."""
    logits = np.asarray(logits, dtype=np.float64)
    _require(logits.ndim == 1 and np.isfinite(logits).all(), "binary logits must have shape (N,) and be finite")
    use = _selection(len(logits), mask)
    y, logits = _labels(labels, len(logits), 2, use), logits[use]
    if not len(y):
        return None
    losses = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    _require(np.isfinite(losses).all(), "binary cross-entropy overflowed")
    return float(losses.mean())


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    p = softmax(logits)
    y = _labels(labels, len(p), p.shape[1], np.ones(len(p), dtype=bool))
    candidates = np.unique(np.r_[np.geomspace(0.1, 10, 121), 1.0])
    losses = [categorical_cross_entropy_from_logits(logits / T, y) for T in candidates]
    best = min(range(len(candidates)), key=lambda i: (losses[i], abs(np.log(candidates[i]))))
    return float(candidates[best])


def elo_expected_score(white_rating, black_rating):
    white, black = np.asarray(white_rating, dtype=float), np.asarray(black_rating, dtype=float)
    _require(np.isfinite(white).all() and np.isfinite(black).all(), "ratings must be finite")
    return np.asarray(0.5 * (1 + np.tanh(np.log(10) * (white - black) / 800)))


def expected_score_metrics(expected_scores, labels, bins=10, *, mask=None):
    scores = np.asarray(expected_scores, dtype=np.float64)
    _require(scores.ndim == 1 and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
             "expected scores must have shape (N,) with finite values in [0,1]")
    use = _selection(len(scores), mask)
    y, scores = _labels(labels, len(scores), 3, use), scores[use]
    actual = np.array([1.0, 0.5, 0.0])[y]
    errors = scores - actual
    return {**_coverage(len(use), use), "mean_squared_error": float((errors ** 2).mean()) if len(y) else None,
            "mean_absolute_error": float(np.abs(errors).mean()) if len(y) else None,
            "reliability": _reliability(scores, actual, bins)}
