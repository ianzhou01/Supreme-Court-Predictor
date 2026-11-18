"""Tests for evaluate() and the two baselines. Synthetic data only, no CSV needed."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluate import always_petitioner, evaluate, previous_terms_majority


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.results_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_evaluate(self, y_true, y_pred, y_score=None, **kwargs):
        kwargs.setdefault("name", "logreg")
        kwargs.setdefault("params", {})
        kwargs.setdefault("owner", "micah")
        if y_score is None:
            y_score = y_pred
        return evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score,
                        results_dir=self.results_dir, **kwargs)

    def test_returns_scores(self) -> None:
        # 3 of 4 right; one missed petitioner win
        scores = self.run_evaluate([1, 1, 0, 0], [1, 0, 0, 0])

        self.assertAlmostEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1"], 2 / 3)
        self.assertAlmostEqual(scores["balanced_accuracy"], 0.75)
        self.assertEqual(scores["n"], 4)
        self.assertEqual((scores["tn"], scores["fp"], scores["fn"], scores["tp"]), (2, 0, 1, 1))

    def test_roc_auc_uses_scores_not_labels(self) -> None:
        # every petitioner win scores above every loss -> perfect ranking
        scores = self.run_evaluate([1, 1, 0, 0], [1, 0, 0, 0], y_score=[0.9, 0.6, 0.4, 0.1])
        self.assertAlmostEqual(scores["roc_auc"], 1.0)

    def test_roc_auc_blank_when_only_one_class(self) -> None:
        scores = self.run_evaluate([1, 1], [1, 1])
        self.assertTrue(pd.isna(scores["roc_auc"]))

    def test_saves_run_to_name_csv(self) -> None:
        self.run_evaluate([1, 0], [1, 1], name="forest", params={"max_depth": 5}, split="val")

        saved = pd.read_csv(self.results_dir / "micah.csv")
        self.assertEqual(len(saved), 1)
        row = saved.iloc[0]
        self.assertEqual(row["model"], "forest")
        self.assertEqual(row["split"], "val")
        self.assertEqual(row["owner"], "micah")
        self.assertEqual(row["params"], '{"max_depth": 5}')
        self.assertAlmostEqual(row["accuracy"], 0.5)
        self.assertIn("timestamp", saved.columns)

    def test_each_run_appends_a_row(self) -> None:
        self.run_evaluate([1, 0], [1, 0], name="a")
        self.run_evaluate([1, 0], [0, 0], name="b")

        saved = pd.read_csv(self.results_dir / "micah.csv")
        self.assertEqual(saved["model"].tolist(), ["a", "b"])

    def test_creates_results_folder(self) -> None:
        nested = self.results_dir / "results"
        evaluate("m", {}, [1], [1], [1], owner="micah", results_dir=nested)
        self.assertTrue((nested / "micah.csv").exists())

    def test_accepts_pandas_series(self) -> None:
        y_true = pd.Series([1, 0, 1], index=[10, 20, 30])
        y_pred = pd.Series([1, 0, 0], index=[10, 20, 30])
        scores = self.run_evaluate(y_true, y_pred)
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)

    def test_rejects_length_mismatch_without_saving(self) -> None:
        with self.assertRaises(ValueError):
            self.run_evaluate([1, 0, 1], [1, 0])
        with self.assertRaises(ValueError):
            self.run_evaluate([1, 0], [1, 0], y_score=[0.5])
        self.assertFalse((self.results_dir / "micah.csv").exists())

    def test_refuses_to_append_to_a_file_with_old_columns(self) -> None:
        (self.results_dir / "micah.csv").write_text("timestamp,model,notes\n2026-01-01,x,\n")
        with self.assertRaises(ValueError):
            self.run_evaluate([1, 0], [1, 0])

    def test_rejects_non_binary_predictions(self) -> None:
        with self.assertRaises(ValueError):
            self.run_evaluate([1, 0], [0.7, 0.2])

    def test_rejects_owner_that_is_not_a_plain_file_name(self) -> None:
        for bad in ("", "../micah", "a/b"):
            with self.subTest(owner=bad), self.assertRaises(ValueError):
                self.run_evaluate([1], [1], owner=bad)


class AlwaysPetitionerTests(unittest.TestCase):
    def test_predicts_one_for_every_case_keeping_index(self) -> None:
        X = pd.DataFrame({"term": [2016, 2017, 2018]}, index=[5, 9, 12])
        pred = always_petitioner(X)
        self.assertEqual(pred.tolist(), [1, 1, 1])
        self.assertEqual(pred.index.tolist(), [5, 9, 12])


def history_frame(wins_by_term):
    """One row per case: {term: [partyWinning, ...]}."""
    rows = [(term, won) for term, outcomes in wins_by_term.items() for won in outcomes]
    return pd.DataFrame(rows, columns=["term", "partyWinning"])


class PreviousTermsMajorityTests(unittest.TestCase):
    def test_uses_majority_of_the_window_before_each_term(self) -> None:
        # Terms 1-2 respondents mostly win; term 3 petitioners all win.
        history = history_frame({1: [0, 0, 1], 2: [0, 0], 3: [1, 1, 1, 1, 1, 1]})
        terms = pd.Series([3, 4], index=["a", "b"])

        pred = previous_terms_majority(history, terms, window=2)

        # term 3 sees terms 1-2 (1 win of 5) -> 0; term 4 sees terms 2-3 (6 of 8) -> 1
        self.assertEqual(pred.tolist(), [0, 1])
        self.assertEqual(pred.index.tolist(), ["a", "b"])

    def test_ignores_the_current_term(self) -> None:
        history = history_frame({1: [0], 2: [1, 1, 1]})
        pred = previous_terms_majority(history, pd.Series([2]), window=10)
        self.assertEqual(pred.tolist(), [0])

    def test_ignores_terms_older_than_the_window(self) -> None:
        history = history_frame({1: [0, 0, 0, 0], 2: [1]})
        pred = previous_terms_majority(history, pd.Series([3]), window=1)
        self.assertEqual(pred.tolist(), [1])

    def test_tie_goes_to_petitioner(self) -> None:
        history = history_frame({1: [0, 1]})
        pred = previous_terms_majority(history, pd.Series([2]), window=1)
        self.assertEqual(pred.tolist(), [1])

    def test_raises_when_a_term_has_no_earlier_cases(self) -> None:
        history = history_frame({5: [1]})
        with self.assertRaises(ValueError):
            previous_terms_majority(history, pd.Series([5]), window=10)


if __name__ == "__main__":
    unittest.main()
