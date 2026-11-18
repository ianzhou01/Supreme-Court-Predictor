"""The two baselines every model has to beat."""

import pandas as pd

from .config import TARGET_COLUMN


def always_petitioner(X):
    """Predict that the petitioner wins every case (1), keeping X's index."""
    return pd.Series(1, index=X.index, name="always_petitioner")


def previous_terms_majority(history, terms, window=10, target=TARGET_COLUMN):
    """Predict each case with the majority outcome of the `window` terms before its term.

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
