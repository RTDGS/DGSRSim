import unittest

from arguments import (
    DEFAULT_SEMANTIC_CLASSES,
    MAX_SEMANTIC_CLASSES,
    MIN_SEMANTIC_CLASSES,
    validate_semantic_class_count,
)


class SemanticClassLimitTests(unittest.TestCase):
    def test_released_default_uses_the_supported_upper_bound(self):
        self.assertEqual(DEFAULT_SEMANTIC_CLASSES, 256)
        self.assertEqual(MAX_SEMANTIC_CLASSES, 256)

    def test_boundary_values_are_accepted(self):
        self.assertEqual(validate_semantic_class_count(MIN_SEMANTIC_CLASSES), 2)
        self.assertEqual(validate_semantic_class_count(MAX_SEMANTIC_CLASSES), 256)

    def test_out_of_range_values_are_rejected(self):
        for value in (0, 1, 257, -1, 256.5, True, "not-an-integer"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_semantic_class_count(value)


if __name__ == "__main__":
    unittest.main()
