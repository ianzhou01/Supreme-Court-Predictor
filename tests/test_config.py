"""Tests for shared configuration."""

import unittest

from src.config import RARE_CATEGORY_THRESHOLD


class ConfigTests(unittest.TestCase):
    def test_rare_category_minimum_count(self) -> None:
        self.assertEqual(RARE_CATEGORY_THRESHOLD, 10)


if __name__ == "__main__":
    unittest.main()
