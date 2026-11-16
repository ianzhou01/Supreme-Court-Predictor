import pandas as pd

from src.config import DERIVED_CATEGORICAL_FEATURES, DERIVED_NUMERIC_FEATURES
from src.features import add_features
from tests.paths import DATA_PATH

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    low_memory=False,
)

df = df[
    df["decisionType"].isin([1, 6, 7])
    & df["partyWinning"].isin([0, 1])
].copy()

unchanged = df.copy()
out = add_features(df)
DERIVED_NUMERIC = DERIVED_NUMERIC_FEATURES
#the circuit features in config come from data.add_circuit_features(), not from here
derived = DERIVED_NUMERIC + ["argument_month"]

added = sorted(set(out.columns) - set(df.columns))
assert added == sorted(derived), f"add_features added {added}"
for name in added:
    assert name in DERIVED_NUMERIC_FEATURES + DERIVED_CATEGORICAL_FEATURES, f"{name} is missing from config.py"
assert df.equals(unchanged), "add_features changed the frame it was given"
assert len(out) == len(df)

#counts taken from the 8,211 filtered rows of SCDB_2026_01
assert out["petitioner_is_us"].sum() == 798
assert out["respondent_is_us"].sum() == 966
assert out["was_reargued"].sum() == 177

assert out[derived].notna().all().all()
assert out[DERIVED_NUMERIC].isin([0, 1]).all().all()
assert set(out["argument_month"].unique()) <= set(range(1, 13)) | {-1}
assert (out["argument_month"] == -1).sum() == 1

#the guard against running after dateArgument and dateRearg are dropped
try:
    add_features(out.drop(columns=["dateArgument"]))
except KeyError:
    pass
else:
    raise AssertionError("expected a KeyError when dateArgument is gone")

print(f"rows: {len(out)}")
for name in DERIVED_NUMERIC:
    print(f"{name}: {out[name].sum()}")
print("argument_month:", out["argument_month"].value_counts().sort_index().to_dict())
print("feature tests passed")
