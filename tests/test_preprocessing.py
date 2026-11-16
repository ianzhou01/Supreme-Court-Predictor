import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.data import add_circuit_features
from src.features import add_features
from src.preprocess import (
    make_onehot_preprocessor,
    make_ordinal_preprocessor,
)
from tests.paths import DATA_PATH

#load supreme court dataset
df = pd.read_csv(DATA_PATH,
encoding="latin-1",
)

#apply our filtering rules
df = df[
    df["decisionType"].isin([1, 6, 7])
    & df["partyWinning"].isin([0, 1])
].copy()

#add the derived features the preprocessors expect
df = add_circuit_features(add_features(df))

#use only training terms
train = df[df["term"] <=2015].copy()

#select model features (original + derived)
X_train = train[FEATURE_COLUMNS].copy()

#match the missing-value convention for this test
X_train["threeJudgeFdc"] = X_train["threeJudgeFdc"].fillna(0)
X_train = X_train.fillna(-1)

#test one-hot encoding
onehot = make_onehot_preprocessor()
X_onehot = onehot.fit_transform(X_train)
print("One-hot encoded shape:", X_onehot.shape)

#test ordinal encoding
ordinal = make_ordinal_preprocessor()
X_ordinal = ordinal.fit_transform(X_train)
print("Ordinal shape:", X_ordinal.shape)

#validate outputs
assert X_onehot.shape[0] == len(X_train)
assert X_ordinal.shape[0] == len(X_train)
assert np.isfinite(X_onehot).all()
assert np.isfinite(X_ordinal).all()
print("All preprocessing tests passed successfully.")

#test unseen categories with example
X_unseen = X_train.iloc[[0]].copy()

#use category not present during training
X_unseen["petitioner"] = 999999
unseen_onehot = onehot.transform(X_unseen)
unseen_ordinal = ordinal.transform(X_unseen)
assert unseen_onehot.shape[0] == 1
assert unseen_ordinal.shape[0] == 1
assert np.isfinite(unseen_onehot).all()
assert np.isfinite(unseen_ordinal).all()
print("Unseen category tests passed successfully.")

#verify rare-category grouping in encoders
onehot_encoder = onehot.named_transformers_["categorical"]
ordinal_encoder = ordinal.named_transformers_["categorical"]

assert any(
    len(indices) > 0
    for indices in onehot_encoder.infrequent_categories_
    if indices is not None
)

assert any(
    len(indices) > 0
    for indices in ordinal_encoder.infrequent_categories_
    if indices is not None
)
print("Rare category grouping tests passed successfully.")