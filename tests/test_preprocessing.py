import unittest
from src.preprocessing import clean_text

class TestPreprocessing(unittest.TestCase):

    def test_clean_text(self):
        sample = "This is an EXCELLENT product!!! :)"
        cleaned = clean_text(sample)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("!", cleaned)
        self.assertTrue("excellent" in cleaned.lower())

if __name__ == '__main__':
    unittest.main()
