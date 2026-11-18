"""Score a model's predictions, log every run to results/<owner>.csv, and the two baselines."""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET_COLUMN

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COLUMNS = ["timestamp", "owner", "model", "params", "split", "n",
           "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
           "tn", "fp", "fn", "tp"]


def evaluate(name, params, y_true, y_pred, y_score, owner, split="val", results_dir=RESULTS_DIR):
    """Score one run and append it as a row of results/<owner>.csv.

    name: short label for the model, e.g. "logreg" or "random forest".
    params: the settings used, as a dict (e.g. {"C": 1}) or a string.
    y_true, y_pred: 0/1 labels (1 = petitioner wins), as lists, arrays or Series.
        Pass class labels, not probabilities: threshold with e.g. (proba >= 0.5).
    y_score: a number per case where higher means "more likely petitioner wins":
        predict_proba(X)[:, 1] for most models, decision_function(X) for SVMs.
        Used only for ROC AUC.
    owner: your name (or "baseline"); it becomes the file name, so letters,
        digits, - and _ only.
    split: which part was scored ("val", or "test" only in task 4.2).

    Precision, recall and F1 treat petitioner wins as the positive class.
    tn/fp/fn/tp are the confusion-matrix counts (true negatives, false positives,
    false negatives, true positives). ROC AUC is blank (NaN) if y_true holds only one class.
    Returns the saved row as a dict.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(owner)):
        raise ValueError(f"owner {owner!r} must be letters, digits, - or _ (it becomes the file name)")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or y_score.shape != y_true.shape:
        raise ValueError(
            f"y_true, y_pred and y_score must be the same length, "
            f"got shapes {y_true.shape}, {y_pred.shape}, {y_score.shape}"
        )
    for label, values in (("y_true", y_true), ("y_pred", y_pred)):
        extra = set(np.unique(values).tolist()) - {0, 1}
        if extra:
            raise ValueError(f"{label} must be 0/1 labels but also has {sorted(extra)}")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score has missing or infinite values")

    if isinstance(params, dict):
        params = json.dumps(params, sort_keys=True, default=str)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "owner": owner,
        "model": name,
        "params": "" if params is None else str(params),
        "split": split,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{owner}.csv"
    new = pd.DataFrame([row], columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != COLUMNS:
            raise ValueError(
                f"{path} has columns {list(old.columns)}, which don't match the current "
                f"evaluate() columns; rename or remove the old file before logging new runs"
            )
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)

    return row


def always_petitioner(X):
    """Baseline 1: predict that the petitioner wins every case (1), keeping X's index."""
    return pd.Series(1, index=X.index, name="always_petitioner")


def previous_terms_majority(history, terms, window=10, target=TARGET_COLUMN):
    """Baseline 2: predict each case with the majority outcome of the `window` terms before its term.

    history: filtered cases with "term" and the target column. Pass the whole
        filtered set (train + validation + test): only terms strictly before each
        case's term are used, so no case sees its own term or anything later.
    terms: the term of each case to predict, e.g. X_val["term"].

    The majority is taken over all cases pooled across the window; a tie goes to
    the petitioner. Returns a 0/1 Series with the same index as `terms`.
    """
    per_term = history.groupby("term")[target].agg(["sum", "count"])

    predictions = {}
    for term in terms.unique():
        prior = per_term.loc[(per_term.index >= term - window) & (per_term.index < term)]
        cases = prior["count"].sum()
        if cases == 0:
            raise ValueError(f"no cases in the {window} terms before {term}")
        predictions[term] = int(2 * prior["sum"].sum() >= cases)

    return terms.map(predictions).astype(int).rename("previous_terms_majority")
