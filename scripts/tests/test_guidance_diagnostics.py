"""Check coverage against the real Q and color-loss selection, without accuracy claims."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
from models.guidance_diagnostics import guidance_report, save_report
from models.learned_color_prior import semantic_reliability, LearnedChromaPrior


class CoverageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p = torch.zeros(1, 3, 8, 12)
        self.p[:, 0, :, :6] = 1; self.p[:, 1, :, 6:] = 1
        self.names = ("sky", "vegetation", "person")

    def test_actual_boundary_exclusion_and_class_counts(self):
        q = semantic_reliability(self.p, boundary_radius=1)
        report = guidance_report(self.p, q, self.names)
        self.assertEqual(report["pixels"], 96)
        for name in self.names[:2]:
            row = report["classes"][name]
            self.assertEqual((row["pixels"], row["threshold_pixels"], row["boundary_pixels"]), (48, 32, 16))
            self.assertEqual(row["boundary_nonzero_pixels"], 0)
        self.assertNotIn("color_loss_eligible_fraction", report)
        self.assertEqual(report["classes"]["person"]["pixels"], 0)

    def test_report_matches_real_fitted_bank_loss_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(dict(format="learned_chroma_gmm_v1", class_names=["sky", "vegetation"],
                classes=dict(sky=dict(status="supported", weights=[1.], means_ab=[[0., 0.]], variances_ab=[[9., 9.]],
                    loss_support=dict(source="group_balanced_train_only", quantile=.99, max_nll=10.)),
                    vegetation=dict(status="unsupported")))))
            bank = LearnedChromaPrior(path, self.names, require_loss_support=True)
            q = semantic_reliability(self.p, boundary_radius=1)
            _, actual = bank.chroma_loss(torch.zeros(1, 3, 8, 12), self.p, q, .5)
            report = guidance_report(self.p, q, self.names, supported=["sky"],
                                     feature_gate=torch.zeros_like(q))
            self.assertAlmostEqual(report["color_loss_eligible_fraction"], float(actual))
            self.assertEqual(report["classes"]["vegetation"]["color_loss_eligible_pixels"], 0)
            self.assertEqual(report["condition_nonzero_fraction"], 0)

    def test_flip_disagreement_reduces_coverage_and_inputs_are_untouched(self):
        p = self.p.clone().requires_grad_(); original = p.detach().clone()
        single = semantic_reliability(p, boundary_radius=1)
        actual = semantic_reliability(p, p.roll(1, 1), boundary_radius=1)
        a = guidance_report(p, single, self.names, supported=["sky", "vegetation"])
        b = guidance_report(p, actual, self.names, supported=["sky", "vegetation"])
        self.assertGreater(a["color_loss_eligible_fraction"], b["color_loss_eligible_fraction"])
        torch.testing.assert_close(p.detach(), original, atol=0, rtol=0)
        self.assertIsNone(p.grad)

    def test_latest_file_replaces_instead_of_growing_and_invalid_scores_are_rejected(self):
        q = semantic_reliability(self.p)
        report = guidance_report(self.p, q, self.names)
        with tempfile.TemporaryDirectory() as tmp:
            save_report(report, tmp); report["scope"] = "next batch"
            path = save_report(report, tmp)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            self.assertEqual(json.loads(path.read_text())["scope"], "next batch")
        for p, quality in ((self.p * .8, q), (self.p, q + float("nan"))):
            with self.assertRaises(ValueError): guidance_report(p, quality, self.names)


if __name__ == "__main__": unittest.main()
