"""Score a model's predictions and log every run to results/<name>.csv."""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COLUMNS = ["timestamp", "model", "split", "n", "accuracy", "balanced_accuracy",
           "precision", "recall", "f1", "notes"]


def evaluate(y_true, y_pred, model, name, split="val", notes="", results_dir=RESULTS_DIR):
    """Score 0/1 predictions and append the run as one row of results/<name>.csv.

    y_true, y_pred: 0/1 labels (1 = petitioner wins), as lists, arrays or Series.
        Pass class labels, not probabilities: threshold with e.g. (proba >= 0.5).
    model: short label for what produced the predictions, e.g. "logreg C=1".
    name: your name; it becomes the file name, so letters, digits, - and _ only.
    split: which part was scored ("val" or "test").
    notes: anything else worth remembering about the run.

    Precision, recall and F1 treat petitioner wins as the positive class.
    Returns the saved row as a dict.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(name)):
        raise ValueError(f"name {name!r} must be letters, digits, - or _ (it becomes the file name)")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}")
    for label, values in (("y_true", y_true), ("y_pred", y_pred)):
        extra = set(np.unique(values).tolist()) - {0, 1}
        if extra:
            raise ValueError(f"{label} must be 0/1 labels but also has {sorted(extra)}")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "split": split,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "notes": notes,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.csv"
    new = pd.DataFrame([row], columns=COLUMNS)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)

    return row
