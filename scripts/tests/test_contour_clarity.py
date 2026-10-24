"""Known-geometry/noise checks and end-to-end clarity contour contracts."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix/tests"))
from models.clarity_semantic_edges import (
    noise_sigma, preprocess_candidates, normal_ridges, soften_ridges,
    canny_vote_candidates, weber_range_candidates, clarity_components,
    make_clarity_semantic_condition, CLARITY_SETTINGS,
)
from models.selective_semantic_edges import selective_components
from models.semantic_edges import FMB_CLASS_NAMES
from models.semantic_contour import SemanticContourGenerator
from models.semantic_color import SemanticColorGenerator
from models.learned_color import LearnedColorGenerator
from models import networks
from test_semantic_contour_training import FakePrior, fixture
from test_learned_color_training import bank_file


def probabilities(name, shape):
    p = np.zeros((15, *shape), np.float32)
    p[FMB_CLASS_NAMES.index(name)] = 1
    return p


class ClarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def noisy_step(self):
        rng = np.random.default_rng(779)
        clean = np.full((128, 160), .3, np.float32)
        clean[25:100, 50:110] = .7
        noisy = np.clip(clean + rng.normal(0, .035, clean.shape), 0, 1).astype(np.float32)
        return clean, noisy

    def test_noise_proxy_tracks_known_awgn_and_vanishes_for_flat_and_ramp(self):
        clean, noisy = self.noisy_step()
        self.assertAlmostEqual(noise_sigma(noisy), .035, delta=.004)
        self.assertEqual(noise_sigma(clean), 0)
        ramp = np.tile(np.arange(64, dtype=np.float32) / 63, (48, 1))
        self.assertLess(noise_sigma(ramp), 1e-7)

    def test_restoration_reduces_known_noise_preserves_step_contrast_and_leaves_input_untouched(self):
        clean, noisy = self.noisy_step(); saved = noisy.copy()
        p = preprocess_candidates(noisy)
        self.assertLess(np.square(p["restored"] - clean).mean(), .6 * np.square(noisy - clean).mean())
        self.assertGreater(float(p["restored"][40:90, 65:95].mean() - p["restored"][40:90, 15:35].mean()), .38)
        np.testing.assert_array_equal(noisy, saved)

    def test_restoration_reduces_far_background_edges_in_a_known_synthetic_scene(self):
        clean, noisy = self.noisy_step()
        restored = preprocess_candidates(noisy)["restored"]
        far = np.ones_like(clean, bool); far[20:105, 45:115] = False
        raw = canny_vote_candidates(noisy); candidate = canny_vote_candidates(restored)
        self.assertLess(np.count_nonzero(candidate["fine_ridges"] & far),
                        .5 * np.count_nonzero(raw["fine_ridges"] & far))
        self.assertEqual(candidate["fine"].max(), 1)

    def test_narrow_encoding_keeps_peak_and_reduces_cross_section_without_rescaling(self):
        ridges = np.zeros((31, 35), bool); ridges[:, 17] = True
        amplitudes = np.ones(ridges.shape, np.float32) * .8
        old = soften_ridges(ridges, amplitudes, width=1)
        new = soften_ridges(ridges, amplitudes)
        np.testing.assert_array_equal(new[:, 17], old[:, 17])
        self.assertLess(new.sum(), .6 * old.sum())
        self.assertLess(new[15, 16], .25 * old[15, 16])

    def test_close_parallel_boundaries_and_real_gap_are_not_merged_by_ridge_extraction(self):
        gray = np.tile(np.arange(64, dtype=np.float32) / 63, (48, 1))
        response = np.zeros_like(gray); response[:, 20] = 1; response[:, 23] = .8
        response[19:25, 20] = 0
        ridges = normal_ridges(gray, response)
        self.assertEqual(np.flatnonzero(ridges[10]).tolist(), [20, 23])
        self.assertFalse(ridges[19:25, 20].any())

    def test_flat_image_cannot_create_semantic_edges_and_weber_zero_is_defined(self):
        for value in (0., .5, 1.):
            gray = np.full((32, 40), value, np.float32)
            p = probabilities("sky", gray.shape)
            p[:, 8:20, 12:28] = probabilities("person", (12, 16))
            maps, _ = clarity_components(gray, p)
            for key in ("clarity_v1", "vote_only", "weber_raw", "weber_enhanced"):
                np.testing.assert_array_equal(maps[key], 0)

    def test_raw_importance_is_identical_and_retained_raw_ridge_peaks_keep_their_floor(self):
        gray = (fixture()[0, 0].numpy() + 1) / 2
        p = probabilities("sky", gray.shape)
        reference = selective_components(gray, p); maps, _ = clarity_components(gray, p)
        for key in ("joint512", "selective_v1", "joint_importance", "importance_gain"):
            np.testing.assert_array_equal(maps[key], reference[key])
        mask = maps["raw_fine_ridges"].astype(bool)
        self.assertTrue(mask.any())
        self.assertTrue(np.all(maps["clarity_v1"][mask] >= .25 * reference["fine"][mask]))
        for key, value in maps.items():
            self.assertEqual(value.dtype, np.float32, key)
            self.assertTrue(np.isfinite(value).all(), key)
            self.assertTrue(np.all((value >= 0) & (value <= 1)), key)

    def test_online_path_matches_full_review_on_crops_and_flipped_geometry(self):
        rng = np.random.default_rng(43)
        for shape in ((32, 40), (48, 48)):
            image = rng.random(shape, dtype=np.float32)
            p = probabilities("person", shape)
            for gray, scores in ((image, p), (image[:, ::-1], p[:, :, ::-1])):
                expected = clarity_components(gray, scores)[0]["clarity_v1"]
                np.testing.assert_array_equal(make_clarity_semantic_condition(gray, scores), expected)

    def test_direct_channel_contract_and_semantic_input_remain_original_tir(self):
        with patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            generator = SemanticContourGenerator(nn.Identity(), "stub", contour_variant="clarity_v1")
        tir = fixture(); actual = generator(tir)
        gray = (tir[0, 0].numpy() + 1) / 2
        np.testing.assert_array_equal(generator.semantic_prior.inputs[0], gray)
        torch.testing.assert_close(actual[:, :1], tir, atol=0, rtol=0)
        self.assertEqual(actual[:, 2:].count_nonzero(), 0)

    def test_semantic_and_learned_color_aux_match_clarity_preview(self):
        with tempfile.TemporaryDirectory() as tmp, patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            backbone = networks.ResnetGenerator(3, 3, ngf=8, n_blocks=1,
                                               norm_layer=networks.get_norm_layer("instance"))
            semantic = SemanticColorGenerator(backbone, "stub", color_condition="semantic", color_width=4,
                                              contour_variant="clarity_v1")
            learned = LearnedColorGenerator(semantic, bank_file(Path(tmp)), width=4, dropout=0.)
            tir = fixture(); gray = (tir[0, 0].numpy() + 1) / 2
            p = torch.from_numpy(semantic.semantic_prior.predict(gray, "cpu"))[None]
            sem_rgb, sem = semantic(tir, return_aux=True, probabilities_override=p)
            rgb, aux = learned(tir, return_aux=True, probabilities_override=p)
            expected = make_clarity_semantic_condition(gray, p[0].numpy())
            np.testing.assert_array_equal(aux["edge"][0, 0].numpy(), expected)
            torch.testing.assert_close(sem["edge"], aux["edge"], atol=0, rtol=0)
            torch.testing.assert_close(rgb, sem_rgb, atol=0, rtol=0)

    def test_clarity_checkpoint_loads_itself_and_rejects_previous_variants(self):
        with patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            new = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant="clarity_v1")
            same = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant="clarity_v1")
            same.load_state_dict(new.state_dict(), strict=True)
            self.assertEqual(new.semantic_metadata["selection"], CLARITY_SETTINGS)
            for variant in ("joint512", "selective_v1"):
                old = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant=variant)
                with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                    old.load_state_dict(new.state_dict(), strict=True)

    def test_rejects_invalid_image_and_ridge_interfaces(self):
        for bad in (np.zeros((5, 7, 3)), np.full((5, 7), np.nan), np.ones((5, 7)) * 2):
            with self.assertRaises(ValueError): preprocess_candidates(bad)
            with self.assertRaises(ValueError): weber_range_candidates(bad)
        with self.assertRaises(ValueError):
            soften_ridges(np.zeros((4, 5), bool), np.zeros((4, 5), np.float32), width=0)


if __name__ == "__main__": unittest.main()
