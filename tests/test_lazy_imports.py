from __future__ import annotations

import importlib
import sys
import unittest


class LazyImportTest(unittest.TestCase):
    def test_pypropel_import_does_not_eagerly_import_plot_stack(self):
        for module_name in [
            "pypropel",
            "pypropel.plot",
            "matplotlib",
            "matplotlib.pyplot",
            "seaborn",
        ]:
            sys.modules.pop(module_name, None)

        pypropel = importlib.import_module("pypropel")

        self.assertNotIn("pypropel.plot", sys.modules)
        self.assertNotIn("matplotlib.pyplot", sys.modules)
        self.assertEqual(pypropel.msa.__name__, "pypropel.msa")


if __name__ == "__main__":
    unittest.main()
