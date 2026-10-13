"""Training gates for geometry, frozen prior, cycle gradients and checkpoints."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import create_model
from models.semantic_contour import SemanticContourGenerator
from models.semantic_edges import make_structure_semantic_contours


class FakePrior:
    def __init__(self, checkpoint):
        with torch.random.fork_rng(devices=[]):
            self.model = nn.Conv2d(3, 15, 1).eval().requires_grad_(False)
        self.info = {"sha256": "ab" * 32}
        self.inputs = []

    def predict(self, gray, device):
        self.inputs.append(gray.copy())
        scores = np.zeros((15, *gray.shape), np.float32)
        scores[8] = np.rint(gray * 255) / 255 * .75
        scores[6] = 1 - scores[8]
        return scores


def fixture():
    image = torch.full((1, 1, 32, 40), -.8)
    image[:, :, 6:25, 9:24] = .8
    image[:, :, 16:27, 30:36] = .5
    return image


def options(folder, model="semantic_contour_cycle_gan", **changes):
    values = dict(
        model=model, isTrain=True, device=torch.device("cpu"),
        checkpoints_dir=str(folder), name=model, direction="BtoA", input_nc=1,
        output_nc=3, lambda_identity=0, ngf=8, ndf=8, netG="resnet_6blocks",
        netD="pixel", n_layers_D=3, norm="instance", no_dropout=True,
        init_type="normal", init_gain=.02, pool_size=2, gan_mode="lsgan",
        lr=.0002, beta1=.5, lambda_A=10., lambda_B=10.,
        condition_mode="soft_saliency", fusion_mode="direct", edge_sigma=1.,
        edge_low_threshold=.08, edge_high_threshold=.16, edge_soft_width=1.,
        soft_high_thresholds=(.12,.16,.20), saliency_sigmas=(.7,1.,1.6),
        saliency_inner_weight=.8, saliency_outer_weight=.2,
        saliency_transition_fraction=24/256, saliency_background_gain=.5,
        saliency_variant="structure_texture", detector_checkpoint=None,
        detector_score_threshold=.2, semantic_checkpoint="stub.pth",
        preprocess="crop", seed=42, continue_train=False, load_iter=0,
        epoch="latest", verbose=False, lr_policy="linear", n_epochs=40,
        n_epochs_decay=40, epoch_count=1,
    )
    values.update(changes)
    (folder / values["name"]).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(**values)


class SemanticTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.prior_patch = patch("models.semantic_contour.FrozenSemanticPrior", FakePrior)
        self.prior_patch.start()
        self.addCleanup(self.prior_patch.stop)

    def test_second_channel_matches_protection_rule_and_third_stays_zero(self):
        generator = SemanticContourGenerator(nn.Identity(), "stub.pth")
        tir = fixture()
        result = generator(tir)
        gray = (tir[0,0].numpy()+1)/2
        scores = generator.semantic_prior.predict(gray, "cpu")
        expected = make_structure_semantic_contours(gray, scores)["joint512"]
        np.testing.assert_array_equal(result[0,1].numpy(), expected)
        torch.testing.assert_close(result[:,:1], tir)
        self.assertEqual(result[:,2:].count_nonzero().item(), 0)

    def test_cropped_flipped_current_input_is_used_without_full_frame_cache(self):
        generator = SemanticContourGenerator(nn.Identity(), "stub.pth")
        tir = fixture()[:,:,4:28,3:35].flip(-1)
        result = generator(tir)
        np.testing.assert_array_equal(generator.semantic_prior.inputs[0], (tir[0,0].numpy()+1)/2)
        self.assertEqual(result.shape, (1,3,24,32))

    def test_batch_images_get_separate_aligned_semantic_maps(self):
        generator = SemanticContourGenerator(nn.Identity(), "stub.pth")
        tir = torch.cat((fixture(), fixture().flip(-1)), 0)
        result = generator(tir)
        self.assertEqual(len(generator.semantic_prior.inputs), 2)
        self.assertEqual(result.shape, (2,3,32,40))
        self.assertFalse(torch.equal(result[0,1], result[1,1]))

    def test_frozen_prior_is_not_reinitialized_trained_or_saved_in_gan(self):
        generator = SemanticContourGenerator(nn.Conv2d(3,3,1), "stub.pth")
        before = generator.semantic_prior.model.weight.clone()
        generator.train()
        generator(fixture()).sum().backward()
        self.assertFalse(generator.semantic_prior.model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in generator.semantic_prior.model.parameters()))
        torch.testing.assert_close(generator.semantic_prior.model.weight, before)
        self.assertEqual(set(generator.state_dict()), {"semantic_signature","backbone.weight","backbone.bias"})
        self.assertGreater(generator.backbone.weight.grad[:,1].abs().sum().item(), 0)

    def test_raw_synthetic_tir_keeps_gradient_while_fixed_edge_is_detached(self):
        generator = SemanticContourGenerator(nn.Conv2d(3,3,1), "stub.pth")
        tir = fixture().requires_grad_()
        rgb, edge = generator(tir, return_condition=True)
        self.assertFalse(edge.requires_grad)
        rgb.sum().backward()
        self.assertGreater(tir.grad.abs().sum().item(), 0)

    def test_changed_candidate_settings_are_rejected(self):
        for changes in ({"edge_soft_width": 2.}, {"soft_high_thresholds": (.12,.16,.20,.24)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SemanticContourGenerator(nn.Identity(), "stub.pth", **changes)

    def test_checkpoint_rejects_a_different_external_semantic_prior(self):
        generator = SemanticContourGenerator(nn.Conv2d(3,3,1), "stub.pth")
        state = copy.deepcopy(generator.state_dict())
        other = SemanticContourGenerator(nn.Conv2d(3,3,1), "stub.pth")
        other.load_state_dict(state)
        state["semantic_signature"][0] ^= 1
        with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
            other.load_state_dict(state)

    def test_full_cyclegan_update_uses_real_and_synthetic_tir_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            opt = options(Path(temporary))
            torch.manual_seed(42)
            model = create_model(opt)
            model.setup(opt)
            tir = fixture()[:,:,:,:32]
            data = {"B":tir,"A":tir.repeat(1,3,1,1),"B_paths":["tir.png"],"A_paths":["rgb.png"]}
            model.set_input(data)
            model.optimize_parameters()
            inputs = model.netG_A.semantic_prior.inputs
            self.assertEqual(len(inputs), 2)
            np.testing.assert_array_equal(inputs[0], (tir[0,0].numpy()+1)/2)
            np.testing.assert_array_equal(inputs[1], np.clip((model.fake_A.detach()[0,0].numpy()+1)/2,0,1))
            self.assertTrue(all(np.isfinite(x) for x in model.get_current_losses().values()))
            self.assertIn("contour_joint", model.get_current_visuals())
            model.eval()
            model.test()
            expected = model.fake_B.clone()
            model.save_networks("latest")
            loaded = create_model(options(Path(temporary), isTrain=False))
            loaded.setup(loaded.opt)
            loaded.eval(); loaded.set_input(data); loaded.test()
            torch.testing.assert_close(loaded.fake_B, expected, atol=1e-6, rtol=1e-6)

    def test_same_seed_initializes_same_trainable_gan_as_c2c_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(42)
            control = create_model(options(root, "conditioned_cycle_gan"))
            control.setup(control.opt)
            torch.manual_seed(42)
            protected = create_model(options(root))
            protected.setup(protected.opt)
            for network in ("netG_A","netG_B","netD_A","netD_B"):
                a = dict(getattr(control,network).named_parameters())
                b = dict(getattr(protected,network).named_parameters())
                self.assertEqual(set(a), set(b))
                for name in a:torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
