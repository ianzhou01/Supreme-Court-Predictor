import pandas as pd

from src.features import (
    add_features,
    DERIVED_CATEGORICAL,
    DERIVED_NUMERIC,
)

df = pd.read_csv(
    "data/SCDB_2026_01_caseCentered_Citation.csv",
    encoding="latin-1",
    low_memory=False,
)

df = df[
    df["decisionType"].isin([1, 6, 7])
    & df["partyWinning"].isin([0, 1])
].copy()

unchanged = df.copy()
out = add_features(df)
derived = DERIVED_NUMERIC + DERIVED_CATEGORICAL

for name in derived:
    assert name in out.columns, f"{name} was not added"
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
