from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DERIVED_CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    RARE_CATEGORY_THRESHOLD,
)

def _feature_groups(extra_categorical=(), extra_numeric=()):
    categories = (list(CATEGORICAL_FEATURES) + list(DERIVED_CATEGORICAL_FEATURES) + list(extra_categorical))
    numbers = (list(NUMERIC_FEATURES) + list(DERIVED_NUMERIC_FEATURES) + list(extra_numeric))
    all_features = categories + numbers
    
    if len(all_features) != len(set(all_features)):
        raise ValueError("A feature is listed more than once")
    
    return categories, numbers

def make_onehot_preprocessor(extra_categorical=(), extra_numeric=()):
    """
    Preprocessing for Logistic Regression, KNN, SVM, and Neural Networks.
    """
    categories, numbers = _feature_groups(extra_categorical, extra_numeric)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
             min_frequency=RARE_CATEGORY_THRESHOLD,
             sparse_output=False),
             categories),
            ("numeric", StandardScaler(), numbers,)
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    
    return preprocessor

def make_ordinal_preprocessor(extra_categorical=(), extra_numeric=()):
    """
    Preprocessing for Decision Trees and tree-based ensemble models.
    """
    categories, numbers = _feature_groups(extra_categorical, extra_numeric)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2, min_frequency=RARE_CATEGORY_THRESHOLD,), categories,),
            ("numeric", "passthrough", numbers,)
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    
    return preprocessor