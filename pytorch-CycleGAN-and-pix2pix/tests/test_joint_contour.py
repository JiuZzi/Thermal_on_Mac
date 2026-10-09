"""Meaningful CPU gates for both label-free single-stage variants."""

import argparse
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import create_model
from models import networks
from models.joint_contour import (
    ContourRefiner, JointContourGenerator, anchor_loss,
    heuristic_anchors, local_map_contrast, transform_view,
)
from models.joint_contour_cycle_gan_model import JointContourCycleGANModel


def test_options(folder, variant="plain", is_train=True, **changes):
    opt = dict(
        model="joint_contour_cycle_gan", isTrain=is_train, device=torch.device("cpu"),
        checkpoints_dir=str(folder), name=variant, dataroot=str(folder / "data"),
        direction="BtoA", input_nc=1, output_nc=3, lambda_identity=0,
        ngf=8, ndf=8, netG="resnet_6blocks", netD="pixel", n_layers_D=3,
        norm="instance", no_dropout=True, init_type="normal", init_gain=0.02,
        pool_size=2, gan_mode="lsgan", lr=0.0002, beta1=0.5, lambda_A=10., lambda_B=10.,
        condition_mode="soft_multi", fusion_mode="direct", edge_sigma=1.,
        edge_low_threshold=0.08, edge_high_threshold=0.16, edge_soft_width=1.,
        soft_high_thresholds=(0.12, 0.16, 0.20), saliency_sigmas=(0.7, 1., 1.6),
        saliency_inner_weight=0.8, saliency_outer_weight=0.2, saliency_transition_fraction=24 / 256,
        saliency_background_gain=0.5, detector_checkpoint=None, detector_score_threshold=0.2,
        joint_variant=variant, contour_width=8, contour_embedding_dim=8,
        contour_epsilon=0.02, contour_residual_limit=6., lambda_contour_anchor=1., lambda_contour_nce=0.1,
        contour_positive_threshold=0.6, contour_background_threshold=0.02,
        contour_gradient_threshold=0.02, contour_exclusion_radius=3,
        contour_max_anchors=16, contour_max_negatives=32, contour_temperature=0.1,
        contour_intensity_jitter=0.05, contour_sampling_seed=2026, contour_intervention="normal",
        joint_smoke=True,
        continue_train=False, epoch="latest", load_iter=0, verbose=False,
        lr_policy="linear", n_epochs=1, n_epochs_decay=1, epoch_count=1, seed=2026,
        batch_size=1, preprocess="crop", crop_size=32, load_size=286,
        no_flip=False, serial_batches=False, max_dataset_size=1,
    )
    opt.update(changes)
    (folder / opt["name"]).mkdir(parents=True, exist_ok=True)
    for split in ("trainA", "trainB"):
        directory = Path(opt["dataroot"]) / split
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "toy.png").exists():
            gray = np.uint8((fixture()[0, 0].numpy() + 1) * 127.5)
            image = Image.fromarray(gray)
            (image.convert("RGB") if split == "trainA" else image).save(directory / "toy.png")
    return SimpleNamespace(**opt)


def fixture():
    tir = torch.full((1, 1, 32, 40), -0.8)
    tir[:, :, 8:24, 8:20] = 0.8
    tir[:, :, 12:18, 30:34] = 0.6
    return tir


def data_fixture():
    tir = fixture()[..., :32]
    return {"B": tir, "A": tir.repeat(1, 3, 1, 1), "A_paths": ["rgb.png"], "B_paths": ["tir.png"]}


def seeded_model(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    model = create_model(opt)
    model.setup(opt)
    return model


class JointContourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fresh_setup_starts_at_clipped_prior_and_keeps_third_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = seeded_model(test_options(Path(temporary)))
            torch.testing.assert_close(model.netG_A.refiner.residual_head.weight,
                                       torch.zeros_like(model.netG_A.refiner.residual_head.weight))
            generator = JointContourGenerator(torch.nn.Identity(), "soft_multi", contour_width=8)
            result = generator(fixture(), return_details=True)
            torch.testing.assert_close(result["edge"], result["prior"].clamp(0.02, 0.98), atol=1e-7, rtol=0)
            self.assertEqual(result["rgb"].shape, (1, 3, 32, 40))
            torch.testing.assert_close(result["rgb"][:, 1:2], result["edge"])
            self.assertEqual(result["rgb"][:, 2:].count_nonzero().item(), 0)

    def test_all_rotations_restore_rectangular_coordinates(self):
        tensor = torch.arange(15).reshape(1, 1, 3, 5)
        for code in range(8):
            torch.testing.assert_close(transform_view(transform_view(tensor, code), code, inverse=True), tensor)

    def test_missing_candidate_at_strong_gradient_is_not_a_negative(self):
        tir = fixture()
        prior = torch.zeros_like(tir)
        pos, neg = heuristic_anchors(tir, prior)
        self.assertFalse(pos.any())
        self.assertFalse(neg[0, 0, 10, 8])
        self.assertTrue(neg[0, 0, 1, 1])

    def test_anchor_loss_ignores_ambiguous_region(self):
        scores = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        prior = torch.zeros_like(scores)
        pos = torch.zeros_like(scores, dtype=torch.bool)
        neg = torch.zeros_like(pos)
        neg[0, 0, 0, 0] = True
        anchor_loss(scores, prior, pos, neg).backward()
        self.assertEqual(scores.grad[0, 0, 1, 1].item(), 0)
        self.assertNotEqual(scores.grad[0, 0, 0, 0].item(), 0)

    def test_nce_alone_reaches_contour_output_head(self):
        torch.manual_seed(26)
        gen = JointContourGenerator(torch.nn.Conv2d(3, 3, 1), "soft_multi", contour_width=8, embedding_dim=8)
        tir = fixture()
        canonical = gen(tir, contour_only=True)
        rotated = gen(transform_view(tir * 0.95, 5), contour_only=True,
                      prior_override=transform_view(canonical["prior"], 5))
        aligned = transform_view(rotated["edge"], 5, inverse=True)
        pos, neg = heuristic_anchors(tir, canonical["prior"])
        loss, active = local_map_contrast(gen.projection(canonical["edge"]), gen.projection(aligned),
                                         pos, neg, torch.Generator().manual_seed(2))
        loss.backward()
        self.assertGreater(active, 0)
        self.assertGreater(gen.refiner.residual_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(gen.projection[0].weight.grad.abs().sum().item(), 0)

    def test_gan_only_gradient_reaches_refiner_and_input(self):
        gen = JointContourGenerator(torch.nn.Conv2d(3, 3, 1), "soft_multi", contour_width=8)
        tir = fixture().requires_grad_(True)
        gen(tir).square().mean().backward()
        self.assertGreater(gen.refiner.residual_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(tir.grad.abs().sum().item(), 0)

    def test_two_variants_share_initial_parameters_and_global_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            plain = seeded_model(test_options(folder, "plain"))
            cl = seeded_model(test_options(folder, "contrastive"))
            for a, b in zip(plain.netG_A.parameters(), cl.netG_A.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(plain.initial_parameters_sha256, cl.initial_parameters_sha256)
            self.assertEqual(plain.training_data_sha256, cl.training_data_sha256)
            for model in (plain, cl):
                model.set_input(data_fixture())
                random.seed(88)
                torch.manual_seed(88)
                numpy_before, python_before, torch_before = np.random.get_state(), random.getstate(), torch.get_rng_state()
                model.forward()
                model.optimizer_G.zero_grad()
                model.set_requires_grad([model.netD_A, model.netD_B], False)
                model.backward_G()
                self.assertEqual(python_before, random.getstate())
                np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
                torch.testing.assert_close(torch_before, torch.get_rng_state(), rtol=0, atol=0)
                if model.opt.joint_variant == "contrastive":
                    self.assertGreater(model.loss_contour_active_anchors, 0)

    def test_optimizer_covers_refiner_and_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = seeded_model(test_options(Path(temporary), "contrastive"))
            optimized = {id(p) for group in model.optimizer_G.param_groups for p in group["params"]}
            self.assertTrue(all(id(p) in optimized for p in model.netG_A.refiner.parameters()))
            self.assertTrue(all(id(p) in optimized for p in model.netG_A.projection.parameters()))
            first = model.netG_A.refiner.residual_head.weight.detach().clone()
            model.set_input(data_fixture())
            model.optimize_parameters()
            self.assertFalse(torch.equal(first, model.netG_A.refiner.residual_head.weight))
            self.assertGreater(model.loss_contour_head_grad, 0)
            self.assertGreater(model.loss_contour_channel_grad, 0)
            # Zero output-head initialization intentionally delays encoder gradients until step 2.
            model.optimize_parameters()
            self.assertGreater(model.netG_A.refiner.enc1[0].weight.grad.abs().sum().item(), 0)

    def test_checkpoint_roundtrip_mismatch_and_epoch_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            opt = test_options(folder, "contrastive")
            model = seeded_model(opt)
            model.set_input(data_fixture())
            model.optimize_parameters()
            model.update_learning_rate()
            model.save_networks(1)
            model.eval()
            with torch.no_grad():
                expected = model.netG_A(fixture(), return_details=True)
            inference_opt = test_options(folder, "contrastive", is_train=False, epoch="1")
            inference = seeded_model(inference_opt)
            inference.eval()
            with torch.no_grad():
                actual = inference.netG_A(fixture(), return_details=True)
            torch.testing.assert_close(expected["edge"], actual["edge"], rtol=0, atol=0)
            torch.testing.assert_close(expected["rgb"], actual["rgb"], rtol=0, atol=0)
            bad_opt = copy.copy(inference_opt)
            bad_opt.joint_variant = "plain"
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                seeded_model(bad_opt)
            # Full training state, including optimizer/scheduler/pools/RNG, resumes at epoch 2.
            model.set_input(data_fixture())
            model.optimize_parameters()
            expected_state = {k: v.clone() for k, v in model.netG_A.state_dict().items()}
            resume_opt = test_options(folder, "contrastive", continue_train=True, epoch="1", epoch_count=2)
            resumed = seeded_model(resume_opt)
            self.assertEqual(resumed.optimizer_G.param_groups[0]["lr"], 0.0001)
            resumed.set_input(data_fixture())
            resumed.optimize_parameters()
            for key, value in expected_state.items():
                torch.testing.assert_close(resumed.netG_A.state_dict()[key], value, rtol=0, atol=0)

    def test_blank_image_produces_inactive_nce_without_nan(self):
        gen = JointContourGenerator(torch.nn.Identity(), "soft_multi", contour_width=8, embedding_dim=8)
        tir = torch.zeros(1, 1, 16, 16)
        d = gen(tir, contour_only=True)
        pos, neg = heuristic_anchors(tir, d["prior"])
        embedding = gen.projection(d["edge"])
        loss, active = local_map_contrast(embedding, embedding, pos, neg, torch.Generator())
        self.assertEqual(active, 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(gen.projection[0].weight.grad).all())

    def test_model_options_select_single_stage_defaults(self):
        parser = argparse.ArgumentParser()
        parser = JointContourCycleGANModel.modify_commandline_options(parser, True)
        opt = parser.parse_args([])
        self.assertEqual((opt.n_epochs, opt.n_epochs_decay), (40, 40))
        self.assertEqual(opt.lambda_identity, 0)
        self.assertEqual(opt.fusion_mode, "direct")


if __name__ == "__main__":
    unittest.main()
