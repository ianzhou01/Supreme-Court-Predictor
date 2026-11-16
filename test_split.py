import pandas as pd

from src.split import split_by_term

df = pd.read_csv(
    "data/SCDB_2026_01_caseCentered_Citation.csv",
    encoding="latin-1",
    low_memory=False,
)

df = df[
    df["decisionType"].isin([1, 6, 7])
    & df["partyWinning"].isin([0, 1])
].copy()
assert len(df) == 8211

# "term" alone is enough to test the split; real callers pass the feature list from config.py
X_train, y_train, X_val, y_val, X_test, y_test = split_by_term(df, ["term"])

# expected sizes and win rates from CLAUDE.local.md
assert (len(X_train), len(X_val), len(X_test)) == (7611, 305, 295)
assert [len(y) for y in (y_train, y_val, y_test)] == [7611, 305, 295]
assert round(y_train.mean() * 100, 1) == 62.8
assert round(y_val.mean() * 100, 1) == 67.2
assert round(y_test.mean() * 100, 1) == 69.2

# boundaries: nothing from a later part leaks into an earlier one
assert X_train["term"].max() == 2015
assert X_val["term"].between(2016, 2020).all()
assert X_test["term"].between(2021, 2025).all()

# every case lands in exactly one part
all_index = X_train.index.union(X_val.index).union(X_test.index)
assert len(all_index) == len(df)

# the target never ends up in X, and y is 0/1 integers
assert "partyWinning" not in X_train.columns
for y in (y_train, y_val, y_test):
    assert y.dtype.kind == "i" and set(y.unique()) <= {0, 1}

# guards: unfiltered targets and target-as-feature are refused
for bad_call in (
    lambda: split_by_term(df.assign(partyWinning=2), ["term"]),
    lambda: split_by_term(df, ["term", "partyWinning"]),
):
    try:
        bad_call()
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError")

print("split tests passed")
