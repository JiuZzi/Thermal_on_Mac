"""Color prior safety, geometry, conditioning gradients and checkpoint gates."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from skimage.color import rgb2lab
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import create_model, networks
from models.semantic_color import SemanticColorGenerator
from models.semantic_color_prior import SemanticColorPrior, rgb_to_lab
from models.semantic_contour import SemanticContourGenerator
from models.semantic_edges import make_structure_semantic_condition
from test_semantic_contour_training import FakePrior, fixture, options as base_options


def scores(label, height=32, width=40):
    result = torch.zeros(1, 15, height, width)
    result[:, label] = 1
    return result


def options(folder, **changes):
    values = dict(color_condition="semantic_color", color_width=4, lambda_color_prior=None,
                  init_semantic_dir=None, init_semantic_epoch="80")
    values.update(changes)
    return base_options(folder, model="semantic_color_cycle_gan", **values)


class ColorPriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.prior = SemanticColorPrior()

    def test_differentiable_lab_matches_skimage_including_black(self):
        generator = torch.Generator().manual_seed(42)
        rgb = torch.rand(2, 3, 8, 9, generator=generator)
        rgb[0, :, 0, 0] = 0
        rgb.requires_grad_()
        actual = rgb_to_lab(rgb)
        expected = rgb2lab(rgb.detach().permute(0, 2, 3, 1).numpy())
        np.testing.assert_allclose(actual.detach().permute(0, 2, 3, 1).numpy(), expected, atol=8e-4)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())

    def test_unsupported_people_have_no_color_constraint(self):
        condition = self.prior(scores(8))
        self.assertEqual(condition["strength"].count_nonzero(), 0)
        self.assertEqual(condition["features"].count_nonzero(), 0)
        rgb = torch.randn(1, 3, 32, 40).tanh().requires_grad_()
        loss = self.prior.chroma_loss(rgb, condition)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())

    def test_semantic_tie_disables_only_color_features(self):
        p = (scores(6) + scores(7)) / 2
        condition = self.prior(p)
        self.assertEqual(condition["strength"].count_nonzero(), 0)
        self.assertEqual(condition["features"].count_nonzero(), 0)
        self.assertEqual(p[:, 6].mean(), .5)

    def test_uncertainty_reduces_absolute_loss_instead_of_cancelling_in_denominator(self):
        rgb = torch.tensor([-.7, -.1, .9])[None, :, None, None].expand(1, 3, 32, 40)
        certain = self.prior.chroma_loss(rgb, self.prior(scores(6)))
        uncertain = self.prior.chroma_loss(rgb, self.prior(.7 * scores(6) + .3 * scores(8)))
        self.assertGreater(certain.item(), 0)
        torch.testing.assert_close(uncertain, certain * .28, atol=1e-6, rtol=1e-5)

    def test_vegetation_blue_penalty_and_color_family_freedom(self):
        condition = self.prior(scores(6))
        for swatch in self.prior.palette_rgb[6]:
            rgb = (swatch[None, :, None, None] * 2 - 1).expand(1, 3, 32, 40)
            self.assertLess(self.prior.chroma_loss(rgb, condition).item(), 1e-6)
        blue = torch.tensor([-.7, -.1, .9])[None, :, None, None].expand(1, 3, 32, 40).clone().requires_grad_()
        loss = self.prior.chroma_loss(blue, condition)
        self.assertGreater(loss.item(), .01)
        loss.backward()
        self.assertGreater(blue.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(blue.grad).all())

    def test_invalid_scores_are_rejected(self):
        for p in (scores(6) * .5, torch.full((1, 15, 2, 2), float("nan")), scores(6)[:, :14]):
            with self.assertRaises(ValueError):
                self.prior(p)


class ColorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        prior_patch = patch("models.semantic_contour.FrozenSemanticPrior", FakePrior)
        prior_patch.start()
        self.addCleanup(prior_patch.stop)

    def generator(self, mode="semantic_color"):
        torch.manual_seed(42)
        backbone = networks.ResnetGenerator(3, 3, ngf=8, n_blocks=1,
                                            norm_layer=networks.get_norm_layer("instance"))
        return SemanticColorGenerator(backbone, "stub.pth", color_condition=mode, color_width=4)

    def test_legacy_initialization_is_exact_and_preserves_tir_edge_zero_contract(self):
        generator = self.generator()
        legacy = SemanticContourGenerator(copy.deepcopy(generator.backbone), "stub.pth")
        generator.initialize_from_semantic(legacy.state_dict())
        seen = []
        handle = generator.backbone.model[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
        try:
            tir = fixture().flip(-1)
            generator.eval(); legacy.eval()
            with torch.no_grad():
                actual, aux = generator(tir, return_aux=True)
                expected = legacy(tir)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(seen[0][:, :1], tir)
            self.assertEqual(seen[0][:, 2:].count_nonzero(), 0)
            gray = (tir[0, 0].numpy() + 1) / 2
            p = generator.semantic_prior.predict(gray, "cpu")
            np.testing.assert_array_equal(aux["edge"][0, 0], make_structure_semantic_condition(gray, p))
        finally:
            handle.remove()

    def test_two_steps_train_condition_branch_and_keep_raw_cycle_gradient(self):
        generator = self.generator()
        optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
        tir = fixture().requires_grad_()
        for _ in range(2):
            optimizer.zero_grad(); tir.grad = None
            rgb = generator(tir)
            rgb.square().mean().backward()
            optimizer.step()
        self.assertGreater(tir.grad.abs().sum().item(), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in generator.condition_encoder.parameters()), 0)
        self.assertFalse(generator.semantic_prior.model.training)
        self.assertTrue(all(p.grad is None for p in generator.semantic_prior.model.parameters()))

    def test_checkpoint_rejects_wrong_mode_or_legacy_silent_load(self):
        generator = self.generator()
        with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
            self.generator("semantic").load_state_dict(generator.state_dict())
        legacy = SemanticContourGenerator(copy.deepcopy(generator.backbone), "stub.pth")
        with self.assertRaises(RuntimeError):
            generator.load_state_dict(legacy.state_dict())

    def test_full_cycle_update_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = create_model(options(root))
            model.setup(model.opt)
            tir = fixture()[:, :, :, :32]
            data = {"B": tir, "A": tir.repeat(1, 3, 1, 1), "B_paths": ["tir.png"], "A_paths": ["rgb.png"]}
            model.set_input(data); model.optimize_parameters()
            self.assertEqual(len(model.netG_A.semantic_prior.inputs), 2)
            self.assertIn("color_prior", model.get_current_losses())
            self.assertTrue(all(np.isfinite(x) for x in model.get_current_losses().values()))
            self.assertGreater(model.loss_color_prior.item(), 0)
            self.assertEqual(model.color_aux["probabilities"].requires_grad, False)
            model.eval(); model.test(); expected = model.fake_B.clone()
            model.save_networks("latest")
            loaded = create_model(options(root, isTrain=False))
            loaded.setup(loaded.opt); loaded.eval(); loaded.set_input(data); loaded.test()
            torch.testing.assert_close(loaded.fake_B, expected, atol=1e-6, rtol=1e-6)

    def test_same_interface_ablation_has_same_initial_backbone_and_finite_zero_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = []
            for mode in ("none", "semantic", "semantic_color"):
                torch.manual_seed(42)
                model = create_model(options(root, name=mode, color_condition=mode))
                model.setup(model.opt)
                models.append(model)
            for other in models[1:]:
                torch.testing.assert_close(models[0].netG_A.backbone.state_dict(), other.netG_A.backbone.state_dict(), atol=0, rtol=0)
                torch.testing.assert_close(models[0].netG_B.state_dict(), other.netG_B.state_dict(), atol=0, rtol=0)
            self.assertEqual(models[0].color_weight, 0)
            self.assertEqual(models[1].color_weight, 0)
            self.assertEqual(models[2].color_weight, 1)


if __name__ == "__main__":
    unittest.main()
