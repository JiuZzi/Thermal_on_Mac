"""Regression and isolation gates for the opt-in R-only recall prototype."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_c2_region_recall import brightness_control, compare_importance
from preview_joint_contours import check_pair
from models.conditioned_cycle_gan_model import ConditionedCycleGANModel
from models.conditioned_generator import ConditionedGenerator
from models.joint_contour_cycle_gan_model import JointContourCycleGANModel
from models.saliency_edges import make_adaptive_saliency_edge, region_recall_importance
from test_joint_contour import seeded_model, test_options


def thermal_image():
    image = np.full((64, 80), .45, np.float32)
    yy, xx = np.indices(image.shape)
    image[((xx - 30) / 6)**2 + ((yy - 30) / 17)**2 <= 1] = .60
    image[48:, :] += .04
    return image


class RegionRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_is_bitwise_identical_to_explicit_original(self):
        args = (thermal_image(), (.12, .16, .20), .5, 1., (.7, 1., 1.6), .5)
        for actual, expected in zip(make_adaptive_saliency_edge(*args),
                                    make_adaptive_saliency_edge(*args, variant="structure_texture")):
            np.testing.assert_array_equal(actual, expected)

    def test_only_R_changes_candidates_and_floor_are_shared(self):
        gray = thermal_image()
        maps, _ = compare_importance(gray)
        actual, importance, fine, coarse = make_adaptive_saliency_edge(
            gray, (.12, .16, .20), .5, 1., (.7, 1., 1.6), .5, variant="region_recall")
        for left, right in ((actual, maps["enhanced"]), (importance, maps["new_R"]),
                            (fine, maps["fine"]), (coarse, maps["coarse"])):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(np.all(actual >= maps["baseline"] - 1e-7))
        self.assertTrue(np.all(actual >= .25 * fine))
        self.assertGreater(float((importance - maps["old_R"]).max()), .01)

    def test_flat_image_and_absent_candidates_are_not_invented(self):
        for level in (0., .5, 1.):
            gray = np.full((32, 32), level, np.float32)
            edge, new_R, _, _ = make_adaptive_saliency_edge(
                gray, (.12, .16, .20), .5, 1., (.7, 1., 1.6), .5, variant="region_recall")
            self.assertEqual(np.count_nonzero(edge), 0)
            np.testing.assert_array_equal(new_R, np.full_like(gray, .32))

    def test_coarse_dominated_route_is_not_sacrificed(self):
        gray = thermal_image()
        base = np.full_like(gray, .32)
        new, _ = region_recall_importance(gray, np.zeros_like(gray), np.ones_like(gray), base)
        np.testing.assert_array_equal(new, base)

    def test_R_bounds_determinism_and_no_input_mutation(self):
        gray = thermal_image()
        fine = np.ones_like(gray)
        coarse = np.zeros_like(gray)
        base = np.full_like(gray, .32)
        copies = [array.copy() for array in (gray, fine, coarse, base)]
        a, cues = region_recall_importance(gray, fine, coarse, base)
        b, _ = region_recall_importance(gray, fine, coarse, base)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a >= base) & (a <= .85)))
        for cue in cues.values():
            self.assertEqual(cue.dtype, np.float32)
            self.assertTrue(np.all((cue >= 0) & (cue <= 1)))
        for value, original in zip((gray, fine, coarse, base), copies):
            np.testing.assert_array_equal(value, original)

    def test_invalid_input_is_rejected(self):
        gray = thermal_image()
        for invalid in (np.full_like(gray, np.nan), np.full_like(gray, -1), np.zeros((4, 4), np.float32)):
            with self.assertRaises(ValueError):
                region_recall_importance(invalid, gray, gray, gray)
        with self.assertRaises(ValueError):
            region_recall_importance(gray, gray, gray, gray, background_gain=0)

    def test_matched_gain_control_has_equal_final_mean_and_keeps_zero_support(self):
        maps, gain = compare_importance(thermal_image())
        self.assertGreaterEqual(gain, 1)
        self.assertAlmostEqual(float(maps["enhanced"].mean(dtype=np.float64)),
                               float(maps["brightness_control"].mean(dtype=np.float64)), places=7)
        np.testing.assert_array_equal(maps["brightness_control"][maps["baseline"] == 0], 0)
        empty, gain = brightness_control(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32))
        self.assertEqual(gain, 1)
        self.assertEqual(np.count_nonzero(empty), 0)
        with self.assertRaises(ValueError):
            brightness_control(np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32))

    def test_generator_batch_dtype_and_reserved_channel_match_formal_variant(self):
        gray = thermal_image()
        tir = torch.from_numpy(np.stack([gray, gray[::-1].copy()])[:, None] * 2 - 1).to(torch.float64)
        generator = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct",
                                         saliency_variant="region_recall")
        edge, reserved = generator.make_condition(tir)
        self.assertEqual(edge.dtype, tir.dtype)
        self.assertEqual(edge.shape, tir.shape)
        self.assertEqual(reserved.count_nonzero().item(), 0)
        for index in range(2):
            processed = np.clip((tir.float().numpy()[index, 0] + 1) / 2, 0, 1)
            expected = compare_importance(processed)[0]["enhanced"]
            np.testing.assert_array_equal(edge[index, 0].numpy(), expected.astype(np.float64))

    def test_new_option_cannot_silently_affect_other_modes_or_overrides(self):
        for mode in ("zero", "canny", "soft_multi", "soft_saliency_position", "soft_object_line"):
            with self.assertRaisesRegex(ValueError, "soft_saliency"):
                ConditionedGenerator(nn.Identity(), mode, fusion_mode="direct", saliency_variant="region_recall")
        generator = ConditionedGenerator(nn.Identity(), "soft_saliency", fusion_mode="direct", saliency_variant="region_recall")
        with self.assertRaisesRegex(ValueError, "override"):
            generator.make_condition(torch.zeros((1, 1, 32, 32)), torch.ones((1, 1, 32, 32)))

    def test_parser_defaults_and_explicit_switch(self):
        parser = ConditionedCycleGANModel.modify_commandline_options(argparse.ArgumentParser(), is_train=False)
        self.assertEqual(parser.parse_args([]).saliency_variant, "structure_texture")
        self.assertEqual(parser.parse_args(["--saliency_variant", "region_recall"]).saliency_variant, "region_recall")

    def test_joint_default_signature_is_unchanged_and_new_variant_is_signed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            original = JointContourCycleGANModel(test_options(folder, condition_mode="soft_saliency", seed=42))
            explicit = JointContourCycleGANModel(test_options(folder, condition_mode="soft_saliency", seed=42,
                                                               saliency_variant="structure_texture"))
            enhanced = JointContourCycleGANModel(test_options(folder, condition_mode="soft_saliency", seed=42,
                                                               saliency_variant="region_recall"))
            self.assertEqual(original.expected_signature, explicit.expected_signature)
            self.assertNotIn("saliency_variant", original.joint_config)
            self.assertNotEqual(original.expected_signature, enhanced.expected_signature)
            self.assertEqual(enhanced.netG_A.saliency_variant, "region_recall")

    def test_joint_preview_rejects_mixed_priors_in_either_order(self):
        def metadata(variant, prior=None):
            config = dict(joint_variant=variant, condition_mode="soft_saliency")
            if prior is not None:
                config["saliency_variant"] = prior
            return dict(joint_config=config, training_data_sha256="same", initial_parameters_sha256="same",
                        training_config={}, scope="quality_unverified", updates=10)
        check_pair([metadata("plain"), metadata("contrastive", "structure_texture")])
        for pair in ([metadata("plain"), metadata("contrastive", "region_recall")],
                     [metadata("plain", "region_recall"), metadata("contrastive")]):
            with self.assertRaisesRegex(ValueError, "saliency_variant"):
                check_pair(pair)

    def test_new_prior_supports_joint_training_without_changing_loss_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = seeded_model(test_options(folder, condition_mode="soft_saliency", seed=42,
                                               contour_sampling_seed=42, saliency_variant="region_recall"))
            tir = torch.from_numpy(thermal_image()[:32, :32][None, None] * 2 - 1)
            model.set_input(dict(B=tir, A=tir.repeat(1, 3, 1, 1), B_paths=["tir.png"], A_paths=["rgb.png"]))
            model.optimize_parameters()
            self.assertTrue(torch.isfinite(model.loss_G))
            self.assertEqual(model.joint_config["saliency_variant"], "region_recall")
            self.assertEqual(model.loss_contour_NCE.item(), 0)


if __name__ == "__main__":
    unittest.main()
