import unittest

import inference_lab


class PackageTest(unittest.TestCase):
    def test_package_exposes_version(self) -> None:
        self.assertEqual(inference_lab.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()

