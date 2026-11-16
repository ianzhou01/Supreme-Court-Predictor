from src.config import FEATURE_COLUMNS, ID_COLUMNS, POST_DECISION_COLUMNS
from src.data import add_circuit_features, load_data
from src.features import add_features
from src.split import split_by_term
from tests.paths import DATA_PATH

df = load_data(DATA_PATH)
assert len(df) == 8211
df = add_circuit_features(add_features(df))

X_train, y_train, X_val, y_val, X_test, y_test = split_by_term(df, FEATURE_COLUMNS)

# expected sizes and win rates
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

# X holds exactly the config features: no target, ID or post-decision columns
for X in (X_train, X_val, X_test):
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert not set(X.columns) & set(ID_COLUMNS + POST_DECISION_COLUMNS)

# y is 0/1 integers
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
