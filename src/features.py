"""Features derived from the raw SCDB columns."""

import pandas as pd

US_PARTY_CODE = 27
MISSING = -1
DATE_FORMAT = "%m/%d/%Y"

SOURCE_COLUMNS = ["petitioner", "respondent", "dateArgument", "dateRearg"]

# The names of the derived features live in config.py
# (DERIVED_NUMERIC_FEATURES, DERIVED_CATEGORICAL_FEATURES).


def add_features(df):
    """Return a copy of df with the four derived features added.

    Run this before handle_missing() drops dateArgument and dateRearg.
    A blank petitioner or respondent counts as not the United States.
    A blank dateArgument gives an argument_month of -1.
    """
    absent = [name for name in SOURCE_COLUMNS if name not in df.columns]
    if absent:
        raise KeyError(
            f"add_features() needs {absent}, so it has to run before those columns are dropped"
        )

    out = df.copy()

    out["petitioner_is_us"] = (out["petitioner"] == US_PARTY_CODE).astype(int)
    out["respondent_is_us"] = (out["respondent"] == US_PARTY_CODE).astype(int)

    argued = pd.to_datetime(out["dateArgument"], format=DATE_FORMAT, errors="coerce")
    out["argument_month"] = argued.dt.month.fillna(MISSING).astype(int)

    reargued = pd.to_datetime(out["dateRearg"], format=DATE_FORMAT, errors="coerce")
    out["was_reargued"] = reargued.notna().astype(int)

    return out
