"""Confirms this environment matches requirements.txt and can train every model type we use.

Run with:  .venv/bin/python check_env.py      (macOS)
           .venv\\Scripts\\python.exe check_env.py   (Windows)
"""
import os
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's startup noise
warnings.filterwarnings("ignore")  # tiny smoke-test fits; convergence warnings don't matter here

ok = True
skip_tf = "--no-tf" in sys.argv  # Intel Macs: TensorFlow isn't installable

# 1. Python version
if sys.version_info[:2] != (3, 13):
    print(f"FAIL  Python {sys.version.split()[0]} (need 3.13.x)")
    ok = False
else:
    print(f"ok    Python {sys.version.split()[0]}")

# 2. Every pinned package at its exact version
req = Path(__file__).with_name("requirements.txt")
for line in req.read_text().splitlines():
    m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
    if not m:
        continue
    name, want = m.groups()
    if skip_tf and name == "tensorflow":
        print("skip  tensorflow (Intel Mac)")
        continue
    try:
        have = version(name)
    except PackageNotFoundError:
        have = "missing"
    status = "ok  " if have == want else "FAIL"
    ok &= have == want
    print(f"{status}  {name} {have}" + ("" if have == want else f" (need {want})"))

# 3. Each model family actually trains
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    rng = np.random.default_rng(42)
    X = rng.random((200, 5))
    y = (X[:, 0] > 0.5).astype(int)
    for model in (LogisticRegression(), RandomForestClassifier(n_estimators=10),
                  SVC(), MLPClassifier(max_iter=300)):
        model.fit(X, y)
    print("ok    scikit-learn: logistic, forest, SVM, MLP all train")

    if not skip_tf:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        net.compile(optimizer="adam", loss="binary_crossentropy")
        net.fit(X, y, epochs=2, verbose=0)
        print("ok    Keras: network trains")
except Exception as e:  # report any failure plainly
    print(f"FAIL  model smoke test: {type(e).__name__}: {e}")
    ok = False

print("\nALL GOOD. Select the 'Python (scotus)' kernel in Jupyter or VS Code."
      if ok else "\nSomething failed. Copy everything above into the group chat.")
sys.exit(0 if ok else 1)