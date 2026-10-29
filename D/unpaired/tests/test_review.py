"""Fusion isolation, unpaired gradients and versioned D-v1/D-v2 contracts."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from D.common import ROOT, read_json, write_json
from D.unpaired.data import ConditionDataset
from D.unpaired.losses import chroma_smoothness, spatial_content
from D.unpaired.model import FormalGenerator, SemanticColorFusion
from D.unpaired.prepare import condition_pipeline_identity, validate_selection
from D.unpaired.protocol import config_from
from D.unpaired.semantics import ConditionBuilder, RGBTeacher
from D.unpaired.tests.test_formal import bank, conditions
from D.tests import test_pipeline as tiny


class TokenPrior(nn.Module):
    feature_channels = 1

    def forward(self, p, q):
        return dict(features=p[:, :1], strength=q)


def fusion(policy="quality_masked"):
    control = SimpleNamespace(controlnet_down_blocks=[nn.Conv2d(1, 1, 1)],
                              controlnet_mid_block=nn.Conv2d(1, 1, 1))
    module = SemanticColorFusion(control, TokenPrior(), width=1, policy=policy, min_quality=.1)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.semantic[0].weight[:, 0].fill_(1)
        module.semantic[2].weight.fill_(1)
        module.color[0].weight.fill_(1)
        module.color[2].weight.fill_(1)
        for layer in [*module.sem_heads, *module.color_heads]:
            layer.weight.fill_(1)
    return module


def fused(module, p, q, *, color=False, size=(8, 8)):
    encoded = module.encode(p, q, size)
    empty = torch.zeros(p.shape[0], 1, *size)
    return module.inject([empty], empty.clone(), encoded,
                         dict(use_semantic=not color, use_color=color, semantic_gate=True))[0][0]


class FusionIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def samples(self):
        p = torch.zeros(1, 15, 8, 8); p[:, 1] = 1
        q = torch.ones(1, 1, 8, 8); q[:, :, 3:5, 3:5] = 0
        changed = p.clone(); changed[:, 0, 3:5, 3:5] = 1; changed[:, 1, 3:5, 3:5] = 0
        return p, changed, q

    def test_untrusted_semantics_cannot_change_a_trusted_neighbor(self):
        p, changed, q = self.samples()
        module = fusion()
        torch.testing.assert_close(fused(module, p, q), fused(module, changed, q), rtol=0, atol=0)
        # Trusted changes must still reach the output; isolation is not a closed branch.
        changed[:, 0, 1:3, 1:3] = 1; changed[:, 1, 1:3, 1:3] = 0
        self.assertGreater(float((fused(module, p, q) - fused(module, changed, q)).abs().max().detach()), 0)

    def test_untrusted_color_tokens_are_isolated_before_convolution(self):
        p, changed, q = self.samples()
        module = fusion()
        torch.testing.assert_close(fused(module, p, q, color=True), fused(module, changed, q, color=True), rtol=0, atol=0)

    def test_masking_precedes_spatial_pooling_and_rejects_low_quality_context(self):
        p, changed, q = self.samples(); q[:, :, 3:5, 3:5] = .05
        module = fusion()
        torch.testing.assert_close(fused(module, p, q, size=(2, 2)),
                                   fused(module, changed, q, size=(2, 2)), rtol=0, atol=0)

    def test_quality_strength_is_applied_once_after_normalized_pooling(self):
        module = fusion()
        p = torch.zeros(1, 15, 8, 8); p[:, 0] = 1
        q = torch.ones(1, 1, 8, 8)
        torch.testing.assert_close(fused(module, p, q * .5), fused(module, p, q) * .5)

    def test_all_unknown_is_finite_and_cannot_generate_residual_bias(self):
        p, changed, q = self.samples(); q.zero_()
        module = fusion()
        with torch.no_grad():
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                    layer.bias.fill_(2)
        actual = fused(module, changed, q)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(int(torch.count_nonzero(actual)), 0)

    def test_legacy_policy_keeps_the_old_context_behavior(self):
        p, changed, q = self.samples()
        module = fusion("legacy")
        self.assertGreater(float((fused(module, changed, q) - fused(module, p, q))[0, 0, 3, 2].detach()), 0)


class InputAndObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_generator_rejects_invalid_probability_and_quality_values(self):
        x, p, q = conditions()
        FormalGenerator._validate(None, x, p, q)
        for bad_p, bad_q in ((p * .5, q), (p, q * float("nan")), (p, q * 1.1)):
            with self.subTest(p_mass=float(bad_p.sum(1).mean()), q_finite=bool(torch.isfinite(bad_q).all())):
                with self.assertRaises(ValueError):
                    FormalGenerator._validate(None, x, bad_p, bad_q)

    def test_synthetic_batch_keeps_each_images_conditions_and_raw_gradient(self):
        builder = object.__new__(ConditionBuilder)
        def compute(image, domain):
            gray = image[:, :, 0].copy()
            p = np.zeros((15, *gray.shape), np.float32); p[int(gray.mean() > 127)] = 1
            return gray, gray.astype(np.float32)/255, p, np.ones_like(gray, dtype=np.float32)
        builder.compute = compute
        tir = torch.stack((torch.full((1, 8, 8), -.5), torch.full((1, 8, 8), .5))).requires_grad_()
        x, p, q = builder.synthetic(tir)
        self.assertEqual(p.shape, (2, 15, 8, 8))
        self.assertTrue(torch.all(p[0, 0] == 1) and torch.all(p[1, 1] == 1))
        self.assertFalse(torch.equal(x[0, 1], x[1, 1]))
        self.assertEqual(int(torch.count_nonzero(x[:, 2])), 0)
        x[:, :1].sum().backward()
        torch.testing.assert_close(tir.grad, torch.full_like(tir, .5))

    def test_content_sampling_never_borrows_another_images_edge_support(self):
        x, p, q = conditions()
        tir = x[:, :1].repeat(2, 1, 1, 1)
        edge = x[:, 1:2].repeat(2, 1, 1, 1); edge[0].zero_()
        rgb = (tir.repeat(1, 3, 1, 1).roll(11, -1)*2-1).requires_grad_()
        actual = spatial_content(rgb, tir, edge, generator=torch.Generator().manual_seed(42))
        expected = spatial_content(rgb[1:], tir[1:], edge[1:], generator=torch.Generator().manual_seed(42)) / 2
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(int(torch.count_nonzero(rgb.grad[0])), 0)
        self.assertGreater(float(rgb.grad[1].abs().sum()), 0)

    def color_sample(self):
        rgb = torch.zeros(1, 3, 8, 8); rgb[:, 0, :, :4] = 1; rgb[:, 2, :, 4:] = 1
        p = torch.zeros(1, 15, 8, 8); p[:, 1] = 1
        return rgb.requires_grad_(), torch.zeros(1, 1, 8, 8), p, torch.ones(1, 1, 8, 8)

    def test_chroma_smoothing_does_not_force_unreliable_regions(self):
        rgb, edge, p, q = self.color_sample()
        legacy = chroma_smoothness(rgb, edge, p)
        actual = chroma_smoothness(rgb, edge, p, q * 0)
        self.assertGreater(float(legacy.detach()), 0)
        self.assertEqual(float(actual.detach()), 0)
        actual.backward(); self.assertEqual(int(torch.count_nonzero(rgb.grad)), 0)

    def test_soft_agreement_and_quality_weaken_the_smoothing_penalty(self):
        rgb, edge, p, q = self.color_sample()
        certain = chroma_smoothness(rgb, edge, p, q)
        ambiguous = p.clone(); ambiguous[:, 1] = .51; ambiguous[:, 2] = .49
        weaker = chroma_smoothness(rgb, edge, ambiguous, q)
        self.assertLess(float(weaker.detach()), float(certain.detach()))
        torch.testing.assert_close(chroma_smoothness(rgb, edge, p, q * .5), certain * .5)
        different = p.clone(); different[:, 1, :, 4:] = 0; different[:, 2, :, 4:] = 1
        self.assertEqual(float(chroma_smoothness(rgb, edge, different, q).detach()), 0)


class VersionAndCacheTests(unittest.TestCase):
    def test_legacy_config_is_not_rewritten_and_v2_is_explicit(self):
        self.assertEqual(config_from(), read_json(ROOT / "D/unpaired/config.json"))
        updated = config_from(ROOT / "D/unpaired/config_review_v2.json")
        self.assertEqual(updated["format"], "D_unpaired_v2")
        self.assertEqual(updated["fusion_policy"], "quality_masked")

    def test_invalid_color_threshold_and_silent_v1_policy_changes_are_rejected(self):
        config = config_from()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.json"
            for changes in ({"color_min_quality": 0}, {"fusion_policy": "quality_masked"}, {"semantic_gate": "false"}):
                write_json(target, dict(config, **changes))
                with self.assertRaises(ValueError): config_from(target)

    def test_duplicate_stems_are_rejected_before_overwriting_outputs(self):
        first = dict(domain="TIR", split="train", image="scene1/frame.png")
        second = dict(first, image="scene2/frame.png")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_selection([first, second])
        validate_selection([first, dict(second, domain="RGB")])

    def test_cache_covers_q_and_mapping_code_and_is_portable_across_line_endings(self):
        initial = condition_pipeline_identity()
        for required in ("D/unpaired/semantics.py", "D/prepare_rgb.py", "pytorch-CycleGAN-and-pix2pix/models/learned_color_prior.py"):
            self.assertIn(required, initial["code_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for relative in initial["code_sha256"]:
                file = base / relative; file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"# fixture\r\nvalue=1\r\n")
            with patch("D.unpaired.prepare.ROOT", base):
                windows = condition_pipeline_identity()
                for file in base.rglob("*.py"):
                    file.write_bytes(file.read_bytes().replace(b"\r\n", b"\n"))
                unix = condition_pipeline_identity()
                self.assertEqual(windows, unix)
                changed = base / "D/unpaired/semantics.py"; changed.write_text("value=2\n")
                updated = condition_pipeline_identity()
                self.assertNotEqual(unix["code_sha256"], updated["code_sha256"])

    def test_dataset_rejects_duplicate_prediction_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frames = [dict(id=f"{i}", stem="same", domain="TIR", role="train") for i in range(2)]
            write_json(folder / "manifest.json", dict(format="D_unpaired_conditions_v1", frames=frames))
            write_json(folder / "status.json", dict(phase="completed", completed=2, total=2))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ConditionDataset(folder, "TIR", "train")


class FullV2ModelTests(unittest.TestCase):
    def setUp(self):
        tiny.TinyPipelineTests.setUpClass()
        self.temp = tempfile.TemporaryDirectory()
        self.path = bank(Path(self.temp.name) / "prior.json")
        self.config = config_from(ROOT / "D/unpaired/config_review_v2.json")
        self.config.update(width=64, height=64, sampling_steps=4, strength=.75, lora_rank=2)
        self.clean_pipeline = copy.deepcopy(tiny.TinyPipelineTests.pipeline)
        self.model = FormalGenerator(tiny.TinyPipelineTests.pipeline, self.config, self.path, {"fixture": "v1"})

    def tearDown(self):
        self.temp.cleanup(); tiny.TinyPipelineTests.tearDownClass()

    def test_legacy_partial_weights_cannot_silently_enter_v2(self):
        legacy_config = {k:v for k,v in self.config.items() if k not in
                         ("fusion_policy", "fusion_min_quality", "chroma_smooth_policy")}
        legacy_config["format"] = "D_unpaired_v1"
        legacy = FormalGenerator(self.clean_pipeline, legacy_config, self.path, {"fixture": "v1"})
        with self.assertRaisesRegex(ValueError, "signature"):
            self.model.import_state(legacy.export_state())

    def test_diagnostics_do_not_change_generation(self):
        x, p, q = conditions(); self.model.eval()
        with torch.no_grad():
            before = self.model(x, p, q)
            self.model.fusion.collect_diagnostics = True
            after = self.model(x, p, q)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(v) for v in self.model.fusion.diagnostics.values()))
        self.assertIn("decoder_saturated_fraction", self.model.output_diagnostics)

    def test_decoder_infinity_is_not_hidden_by_output_clamping(self):
        x, p, q = conditions()
        invalid = SimpleNamespace(sample=torch.full((1, 3, 64, 64), float("inf")))
        with patch.object(self.model.vae, "decode", return_value=invalid):
            with self.assertRaisesRegex(FloatingPointError, "before output clamping"):
                self.model(x, p, q)

    def test_conditioning_only_prior_does_not_require_unused_loss_support(self):
        contents = read_json(self.path)
        del contents["classes"]["road"]["loss_support"]
        write_json(self.path, contents)
        config = copy.deepcopy(self.config); config["loss_weights"]["color_prior"] = 0
        model = FormalGenerator(self.clean_pipeline, config, self.path, {"fixture": "v1"})
        x, p, q = conditions()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(x, p, q)).all())

    def test_complete_unpaired_update_and_replay_have_finite_gradients(self):
        from D.unpaired.engine import UnpairedEngine
        x, p, q = conditions()
        class Builder:
            def synthetic(self, tir):
                return torch.cat(((tir+1)/2, x[:, 1:2], x[:, 2:3]), 1), p, q
        class Teacher(nn.Module):
            def forward(self, rgb, side=256):
                return rgb.mean(1, keepdim=True).repeat(1, 15, 1, 1).softmax(1)
        reverse = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.Tanh())
        critics = nn.ModuleDict(dict(RGB=nn.Conv2d(3, 1, 1), TIR=nn.Conv2d(1, 1, 1)))
        config = dict(self.config, replay_every=1)
        engine = UnpairedEngine(self.model, reverse, critics, Builder(), Teacher(), config, device="cpu")
        before = self.model.export_state()
        sample = dict(x=x, p=p, q=q, rgb=torch.rand(1, 3, 64, 64)*2-1)
        applied, values = engine.update(sample, sample, 0, 10, diagnostics=True)
        self.assertTrue(applied)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(v)) for v in values.values()))
        weighted = sum(value for key, value in values.items() if key.startswith("weighted_"))
        torch.testing.assert_close(values["G_total"], weighted)
        self.assertEqual(float(values["grad_C_zero"]), 0)
        self.assertGreater(float(values["grad_semantic_adapter"]), 0)
        self.assertGreater(float(values["grad_color_adapter"]), 0)
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in self.model.export_state().items()))


class FrozenTeacherTests(unittest.TestCase):
    def test_real_segformer_mapping_preserves_rgb_gradient_with_frozen_weights(self):
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        from D.prepare_rgb import CITY_TO_FMB, NAMES
        torch.set_num_threads(1)
        teacher = object.__new__(RGBTeacher); nn.Module.__init__(teacher)
        config = SegformerConfig(num_encoder_blocks=2, depths=[1, 1], hidden_sizes=[8, 16],
            num_attention_heads=[1, 2], sr_ratios=[4, 2], patch_sizes=[3, 3], strides=[2, 2],
            decoder_hidden_size=16, num_labels=19)
        teacher.model = SegformerForSemanticSegmentation(config).eval().requires_grad_(False)
        teacher.register_buffer("mean", torch.full((1, 3, 1, 1), .5))
        teacher.register_buffer("std", torch.full((1, 3, 1, 1), .25))
        mapping = torch.zeros(15, 19)
        for index, name in enumerate(CITY_TO_FMB):
            mapping[NAMES.index(CITY_TO_FMB[name]), index] = 1
        teacher.register_buffer("mapping", mapping)
        rgb = (torch.rand(1, 3, 64, 64)*2-1).requires_grad_()
        probabilities = teacher(rgb, side=32)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1, 64, 64))
        (-probabilities[:, 1].clamp_min(1e-7).log().mean()).backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertGreater(float(rgb.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))


if __name__ == "__main__":
    unittest.main()
