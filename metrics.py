from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ConfusionMatrix:
    tn: int
    fp: int
    fn: int
    tp: int

    def as_matrix(self) -> list[list[int]]:
        return [[int(self.tn), int(self.fp)], [int(self.fn), int(self.tp)]]

    @property
    def total(self) -> int:
        return int(self.tn + self.fp + self.fn + self.tp)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    loss: float
    acc: float
    roc_auc: float
    pr_auc: float
    threshold: float
    cm: ConfusionMatrix
    precision: float
    recall: float
    f1: float
    specificity: float
    npv: float
    balanced_acc: float


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return float("nan")
    return float(num) / float(den)


def roc_auc_mann_whitney(probs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [(float(p), float(y)) for p, y in zip(probs, ys)]
    n_pos = sum(1 for _, y in pairs if y >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs.sort(key=lambda t: float(t[0]))
    ranks = [0.0] * len(pairs)
    i = 0
    r = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and float(pairs[j][0]) == float(pairs[i][0]):
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        r += (j - i)
        i = j

    sum_ranks_pos = 0.0
    for (_, y), rk in zip(pairs, ranks):
        if float(y) >= 0.5:
            sum_ranks_pos += float(rk)

    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u) / float(n_pos * n_neg)


def average_precision(probs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = sorted(((float(p), float(y)) for p, y in zip(probs, ys)), key=lambda t: float(t[0]), reverse=True)
    positives_total = sum(1 for _, y in pairs if float(y) >= 0.5)
    if positives_total == 0:
        return float("nan")

    tp = 0
    fp = 0
    precision_sum = 0.0
    for _, y in pairs:
        if float(y) >= 0.5:
            tp += 1
            precision_sum += tp / max(1, tp + fp)
        else:
            fp += 1
    return float(precision_sum) / float(positives_total)


class BinaryClassificationEvaluator:
    def __init__(self, *, criterion: torch.nn.Module, threshold: float = 0.5):
        self.criterion = criterion
        self.threshold = float(threshold)

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> BinaryClassificationMetrics:
        model.eval()

        total_loss = 0.0
        total = 0
        correct = 0

        tn = fp = fn = tp = 0
        all_probs: list[float] = []
        all_y: list[float] = []

        thr = float(self.threshold)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = self.criterion(logits, y)

            bs = int(x.shape[0])
            total_loss += float(loss.item()) * bs
            total += bs

            probs = torch.sigmoid(logits)
            pred_pos = probs >= thr
            y_pos = y >= 0.5

            correct += int((pred_pos == y_pos).sum().item())
            tp += int((pred_pos & y_pos).sum().item())
            tn += int((~pred_pos & ~y_pos).sum().item())
            fp += int((pred_pos & ~y_pos).sum().item())
            fn += int((~pred_pos & y_pos).sum().item())

            all_probs.extend(probs.detach().flatten().to("cpu").tolist())
            all_y.extend(y.detach().flatten().to("cpu").tolist())

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        roc_auc = roc_auc_mann_whitney(all_probs, all_y) if all_probs else float("nan")
        pr_auc = average_precision(all_probs, all_y) if all_probs else float("nan")

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        npv = _safe_div(tn, tn + fn)

        f1 = float("nan")
        if precision == precision and recall == recall and (precision + recall) > 0:  # not-NaN and denom > 0
            f1 = 2.0 * precision * recall / (precision + recall)

        balanced_acc = float("nan")
        if recall == recall and specificity == specificity:
            balanced_acc = 0.5 * (recall + specificity)

        cm = ConfusionMatrix(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        return BinaryClassificationMetrics(
            loss=float(avg_loss),
            acc=float(acc),
            roc_auc=float(roc_auc),
            pr_auc=float(pr_auc),
            threshold=float(thr),
            cm=cm,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            specificity=float(specificity),
            npv=float(npv),
            balanced_acc=float(balanced_acc),
        )

