"""Known geometry and integration contracts for bounded R-only recall."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for folder in ("scripts", "pytorch-CycleGAN-and-pix2pix", "pytorch-CycleGAN-and-pix2pix/tests"):
    sys.path.insert(0, str(ROOT/folder))
from models.recall_semantic_edges import (
    RECALL_SETTINGS, recall_importance, recall_components, make_recall_semantic_condition,
)
from models.semantic_contour import SemanticContourGenerator
from models.semantic_color import SemanticColorGenerator
from models.learned_color import LearnedColorGenerator
from models.semantic_edges import FMB_CLASS_NAMES
from models import networks
from test_semantic_contour_training import FakePrior, fixture
from test_learned_color_training import bank_file


def scores(name, shape):
    p = np.zeros((15, *shape), np.float32)
    p[FMB_CLASS_NAMES.index(name)] = 1
    return p


class ImportanceRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def silhouette(self):
        image = np.full((72, 88), .3, np.float32)
        yy, xx = np.indices(image.shape)
        image[(xx-44)**2 + (yy-16)**2 <= 25] = .75
        image[22:45, 39:49] = .75
        image[28:33, 32:55] = .75
        image[45:63, 39:42] = .75; image[45:63, 46:49] = .75
        return image

    def test_coarse_only_and_dense_clutter_cannot_be_weakened_by_r_increase(self):
        image = self.silhouette(); shape = image.shape
        base = np.full(shape, .35, np.float32)
        fine, coarse = np.full(shape, .2, np.float32), np.full(shape, .8, np.float32)
        actual, _ = recall_importance(image, fine, coarse, base, np.zeros(shape, np.float32))
        np.testing.assert_array_equal(actual, base)
        actual, _ = recall_importance(image, coarse, fine, base, np.ones(shape, np.float32))
        np.testing.assert_array_equal(actual, base)

    def test_existing_object_edges_gain_support_even_when_semantics_says_sky(self):
        image = self.silhouette(); p = scores("sky", image.shape); saved = p.copy()
        maps, _ = recall_components(image, p)
        ridges = maps["clarity_v1_fine_ridges"].astype(bool)
        self.assertTrue(ridges.any())
        self.assertGreater(float(maps["recall_v1"][ridges].mean()),
                           float(maps["clarity_v1"][ridges].mean()))
        np.testing.assert_array_equal(p, saved)

    def test_monotone_bounded_output_keeps_candidates_width_and_floor_identical(self):
        rng = np.random.default_rng(831)
        for image in (self.silhouette(), rng.random((40, 48), dtype=np.float32)):
            maps, _ = recall_components(image, scores("sky", image.shape))
            self.assertTrue(np.all(maps["recall_v1"] >= maps["clarity_v1"]-1e-7))
            self.assertTrue(np.all(maps["recall_importance"] >= maps["joint_importance"]-1e-7))
            self.assertLessEqual(maps["recall_importance"].max(), .85)
            no_advantage = maps["clarity_v1_fine"] <= .5*maps["clarity_v1_coarse"]
            np.testing.assert_array_equal(maps["recall_importance"][no_advantage],
                                          maps["joint_importance"][no_advantage])
            for value in maps.values():
                self.assertTrue(np.isfinite(value).all())
                self.assertTrue(np.all((value >= 0) & (value <= 1)))
            self.assertTrue(np.all(maps["recall_v1"] >= maps["preserved_ridge_floor"]))

    def test_flat_image_with_semantic_person_patch_cannot_invent_an_edge(self):
        image = np.full((40, 48), .4, np.float32)
        p = scores("sky", image.shape); p[:, 10:30, 15:35] = scores("person", (20, 20))
        maps, _ = recall_components(image, p)
        np.testing.assert_array_equal(maps["recall_v1"], 0)
        np.testing.assert_array_equal(maps["recall_importance"], maps["joint_importance"])

    def test_preview_and_online_recall_agree_on_cropped_and_flipped_geometry(self):
        image = self.silhouette()[8:64, 12:76]
        for gray in (image, image[:, ::-1]):
            p = scores("person", gray.shape)
            expected = recall_components(gray, p)[0]["recall_v1"]
            np.testing.assert_array_equal(make_recall_semantic_condition(gray, p), expected)

    def test_one_semantic_pass_original_tir_and_reserved_zero_channel(self):
        with patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            generator = SemanticContourGenerator(nn.Identity(), "stub", contour_variant="recall_v1")
        tir = fixture(); actual = generator(tir)
        self.assertEqual(len(generator.semantic_prior.inputs), tir.shape[0])
        np.testing.assert_array_equal(generator.semantic_prior.inputs[0], (tir[0, 0].numpy()+1)/2)
        torch.testing.assert_close(actual[:, :1], tir, rtol=0, atol=0)
        self.assertEqual(actual[:, 2:].count_nonzero(), 0)

    def test_semantic_and_learned_colour_share_recall_without_changing_zero_branch(self):
        with tempfile.TemporaryDirectory() as tmp, patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            backbone = networks.ResnetGenerator(3, 3, ngf=8, n_blocks=1,
                                               norm_layer=networks.get_norm_layer("instance"))
            semantic = SemanticColorGenerator(backbone, "stub", color_condition="semantic", color_width=4,
                                               contour_variant="recall_v1")
            learned = LearnedColorGenerator(semantic, bank_file(Path(tmp)), width=4, dropout=0.)
            tir = fixture(); gray = (tir[0, 0].numpy()+1)/2
            p = torch.from_numpy(semantic.semantic_prior.predict(gray, "cpu"))[None]
            sem_rgb, sem = semantic(tir, return_aux=True, probabilities_override=p)
            rgb, aux = learned(tir, return_aux=True, probabilities_override=p)
            np.testing.assert_array_equal(aux["edge"][0, 0].numpy(), make_recall_semantic_condition(gray, p[0].numpy()))
            torch.testing.assert_close(sem["edge"], aux["edge"], rtol=0, atol=0)
            torch.testing.assert_close(rgb, sem_rgb, rtol=0, atol=0)

    def test_recipe_signature_isolated_from_all_older_variants(self):
        with patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            generator = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant="recall_v1")
            same = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant="recall_v1")
            same.load_state_dict(generator.state_dict(), strict=True)
            self.assertEqual(generator.semantic_metadata["selection"], RECALL_SETTINGS)
            for variant in ("joint512", "selective_v1", "clarity_v1"):
                old = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant=variant)
                # PyTorch may copy buffers before reporting load errors. Use a
                # snapshot of the untouched old checkpoint for the reverse test.
                old_state = {key: value.clone() for key, value in old.state_dict().items()}
                with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                    old.load_state_dict(generator.state_dict(), strict=True)
                with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                    generator.load_state_dict(old_state, strict=True)
                # Restore a valid new signature before checking the next pair.
                generator = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant="recall_v1")


if __name__ == "__main__": unittest.main()
