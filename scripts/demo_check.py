"""Independent acceptance check for the synthetic public-repository issue."""

import sys
import unittest

sys.path.insert(0, "/work/src")
from sample.simple import add_one  # noqa: E402


class AddOneTests(unittest.TestCase):
    def test_default_and_explicit_arguments(self):
        self.assertEqual(add_one(7), 8)
        self.assertEqual(add_one(-2), -1)
        self.assertEqual(add_one(), 1)


if __name__ == "__main__":
    unittest.main()
