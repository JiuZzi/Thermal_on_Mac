"""Seeded runpy startup must resolve report code without metric recomputation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


class EvaluationEntrypointTests(unittest.TestCase):
    def test_runpy_resolves_sibling_renderer_from_another_working_directory(self):
        code = (
            "import runpy; from pathlib import Path; "
            f"runpy.run_path({str(SCRIPTS / 'evaluate_distribution.py')!r}); "
            "import render_metrics_figure as report; "
            f"assert Path(report.__file__).resolve() == Path({str(SCRIPTS / 'render_metrics_figure.py')!r}); "
            "assert callable(report.render_metric_report)"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=directory, text=True,
                capture_output=True, env={**os.environ, "MPLBACKEND": "Agg"},
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
