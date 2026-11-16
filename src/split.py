"""Split the filtered SCDB cases into training, validation and test parts by term.

Note: The split is always by term (the year the case was heard), never random:
a random shuffle would let the model learn from cases decided after the
ones it is being tested on.
"""

from .config import (
    TARGET_COLUMN,
    TEST_END_TERM,
    TEST_START_TERM,
    TRAIN_END_TERM,
    VALIDATION_END_TERM,
    VALIDATION_START_TERM,
)


def split_by_term(df, feature_columns, target=TARGET_COLUMN):
    """Docs: This function splits the DataFrame into training, validation, and test sets based on the term (year) of each case.

    df: the filtered cases (decisionType in {1, 6, 7}, partyWinning in {0, 1}),
        with derived features already added. It needs a "term" column and the target column.
    feature_columns: the columns to put in X. Pass the feature list from config.py,
        so ID and post-decision columns never reach the model.
    target: the column to put in y (default "partyWinning").

    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    X parts are DataFrames of feature_columns; y parts are int Series (1 = petitioner wins).
    Row index is kept from df, so rows can be matched back to the original cases.
    Prints how many cases land in each part and the petitioner win rate of each.
    """
    needed = ["term", target]
    absent = [name for name in needed + list(feature_columns) if name not in df.columns]
    if absent:
        raise KeyError(f"split_by_term() can't find these columns: {absent}")
    if target in feature_columns:
        raise ValueError(f"{target} is the answer, so it can't also be a feature")

    bad_target = set(df[target].unique()) - {0, 1}
    if bad_target:
        raise ValueError(f"{target} should only be 0 or 1, but also has {sorted(bad_target)}; filter first")

    term = df["term"]
    in_train = term <= TRAIN_END_TERM
    in_val = term.between(VALIDATION_START_TERM, VALIDATION_END_TERM)
    in_test = term.between(TEST_START_TERM, TEST_END_TERM)

    # Every case must land in exactly one part, so none are silently lost.
    leftover = df.loc[~(in_train | in_val | in_test), "term"]
    if len(leftover):
        raise ValueError(f"{len(leftover)} cases have terms outside the split: {sorted(leftover.unique())}")

    parts = []
    for name, mask in [("train", in_train), ("val", in_val), ("test", in_test)]:
        X = df.loc[mask, list(feature_columns)]
        y = df.loc[mask, target].astype(int)
        print(f"{name:>5}: {len(y):>5,} cases, terms {term[mask].min()}-{term[mask].max()}, "
              f"petitioner win rate {y.mean():.1%}")
        parts += [X, y]

    return tuple(parts)
