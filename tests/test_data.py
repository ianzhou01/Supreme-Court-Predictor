"""Tests for SCDB loading and missing-value handling."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from src.data import add_circuit_features, handle_missing, load_data


class LoadDataTests(unittest.TestCase):
    def test_load_data_filters_rows_and_reports_counts(self) -> None:
        rows = pd.DataFrame(
            {
                "caseName": ["Peña", "A", "B", "C", "D"],
                "decisionType": [1, 6, 7, 2, 1],
                "partyWinning": [0, 1, None, 1, 1],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            rows.to_csv(path, index=False, encoding="latin-1")
            output = io.StringIO()
            with redirect_stdout(output):
                result = load_data(path)

        self.assertEqual(result["caseName"].tolist(), ["Peña", "A", "D"])
        self.assertIn("Number of rows loaded: 5", output.getvalue())
        self.assertIn("decisionType filter: 4", output.getvalue())
        self.assertIn("partyWinning filter: 3", output.getvalue())


class HandleMissingTests(unittest.TestCase):
    def test_handle_missing_drops_columns_and_fills_values(self) -> None:
        raw = pd.DataFrame(
            {
                "caseId": ["a", "b"],
                "partyWinning": [0, 1],
                "caseDisposition": [2, 3],
                "chief": ["Burger", "Burger"],
                "threeJudgeFdc": [None, 1],
                "issue": [None, 10010],
                "adminAction": [42, None],
            }
        )

        clean = handle_missing(raw)

        self.assertNotIn("caseId", clean.columns)
        self.assertNotIn("partyWinning", clean.columns)
        self.assertNotIn("caseDisposition", clean.columns)
        self.assertNotIn("chief", clean.columns)
        self.assertEqual(clean["threeJudgeFdc"].tolist(), [0, 1])
        self.assertEqual(clean["issue"].tolist(), [-1, 10010])
        self.assertEqual(clean["adminAction"].tolist(), [42, -1])
        self.assertFalse(clean.isna().any().any())
        self.assertTrue(raw.isna().any().any())

    def test_handle_missing_allows_absent_drop_columns(self) -> None:
        raw = pd.DataFrame({"issueArea": [None, 1]})
        result = handle_missing(raw)
        self.assertEqual(result["issueArea"].tolist(), [-1, 1])


class CircuitFeatureTests(unittest.TestCase):
    def test_add_circuit_features_maps_known_and_unknown_courts(self) -> None:
        raw = pd.DataFrame(
            {
                "caseSource": [21, 22, 32, 1, None, 999],
                "caseOrigin": [31, 30, 440, 4, 19, 999],
            }
        )

        mapped = add_circuit_features(raw)

        self.assertEqual(
            mapped["caseSourceCircuit"].tolist(), [1, 2, 12, 13, 0, 0]
        )
        self.assertEqual(
            mapped["caseOriginCircuit"].tolist(), [11, 10, 12, 14, 0, 0]
        )
        self.assertNotIn("caseSourceCircuit", raw.columns)
        self.assertNotIn("caseOriginCircuit", raw.columns)

    def test_add_circuit_features_requires_both_court_columns(self) -> None:
        with self.assertRaisesRegex(KeyError, "caseOrigin"):
            add_circuit_features(pd.DataFrame({"caseSource": [21]}))


if __name__ == "__main__":
    unittest.main()
