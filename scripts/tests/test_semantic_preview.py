"""Contract checks for frozen semantic review, separate from quality evidence."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from semantic_preview.checkpoint import validate_state_dict, load_checkpoint, sha256_file
from semantic_preview.inference import preprocess, predict, predict_probabilities, MEAN, STD
from semantic_preview.model import ConvModule
from semantic_preview.review import render_crop, write_page, score_maps
import preview_semantics


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        logits = torch.zeros(1, 15, 2, 3)
        logits[0, 8] = torch.tensor([[0., 2., 7.], [-2., 3., 1.]])
        self.register_buffer("logits", logits)

    def forward(self, image):
        return self.logits


class InputLogits(nn.Module):
    def forward(self, image):
        values = F.adaptive_avg_pool2d(image[:, :1], (4, 5))
        return torch.cat([values * (i - 7) * .1 for i in range(15)], 1)


class SemanticPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preprocess_matches_author_grayscale_resize_and_channel_normalization(self):
        gray = np.arange(39 * 51, dtype=np.uint8).reshape(39, 51)
        actual, shape = preprocess(gray, 64)
        rgb = np.stack([gray] * 3, -1)
        # Independent direct expression of the selected author's operations.
        resized = np.asarray(Image.fromarray(rgb).resize((int(64/39*51), int(64/39*39)), Image.Resampling.BILINEAR))
        reference = np.empty_like(resized, dtype=np.float32)
        for c in range(3):
            reference[..., c] = (resized[..., c] - MEAN[c]) / STD[c]
        np.testing.assert_array_equal(actual.numpy()[0], reference.transpose(2, 0, 1))
        self.assertEqual(shape, resized.shape[:2])

    def test_author_preprocess_does_not_shrink_large_images(self):
        gray = np.zeros((66, 71), np.uint8)
        tensor, shape = preprocess(gray, 64)
        self.assertEqual(shape, (66, 71))
        self.assertEqual(tuple(tensor.shape), (1, 3, 66, 71))

    def test_preprocess_rejects_rgb_or_16bit_instead_of_silent_conversion(self):
        for gray in (np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint16), np.zeros((0, 10), np.uint8)):
            with self.assertRaises(ValueError):
                preprocess(gray)

    def test_logits_restored_before_softmax_on_original_rectangular_coordinates(self):
        model = FixedLogits().eval().requires_grad_(False)
        gray = np.zeros((11, 17), np.uint8)
        output = predict(model, gray, "cpu", short_side=32)
        reference = F.interpolate(model.logits, size=gray.shape, mode="bilinear", align_corners=True).softmax(1)[0].numpy()
        np.testing.assert_array_equal(output["probabilities"], reference)
        wrong_order = F.interpolate(model.logits.softmax(1), size=gray.shape, mode="bilinear", align_corners=True)[0].numpy()
        self.assertGreater(float(np.abs(reference-wrong_order).max()), .01)
        self.assertEqual(output["labels"].shape, gray.shape)

    def test_probabilities_only_match_review_without_diagnostic_entropy(self):
        gray = np.arange(11 * 17, dtype=np.uint8).reshape(11, 17)
        for flip_tta in (False, True):
            model = InputLogits().eval().requires_grad_(False)
            expected = predict(model, gray, "cpu", short_side=32,
                               flip_tta=flip_tta)["probabilities"]
            with patch("semantic_preview.inference.np.log",
                       side_effect=AssertionError("Training must not compute entropy")):
                actual = predict_probabilities(model, gray, "cpu", short_side=32,
                                               flip_tta=flip_tta)
            self.assertIsInstance(actual, np.ndarray)
            np.testing.assert_array_equal(actual, expected)

    def test_flip_probe_inverts_coordinates_and_averages_class_responses(self):
        model = InputLogits().eval().requires_grad_(False)
        gray = np.arange(11 * 17, dtype=np.uint8).reshape(11, 17)
        output = predict(model, gray, "cpu", short_side=32, flip_tta=True)
        tensor, _ = preprocess(gray, 32)
        a = F.interpolate(model(tensor), size=gray.shape, mode="bilinear", align_corners=True).softmax(1)
        b = F.interpolate(model(tensor.flip(-1)), size=gray.shape, mode="bilinear", align_corners=True).flip(-1).softmax(1)
        np.testing.assert_array_equal(output["probabilities"], ((a+b)*.5)[0].numpy())
        np.testing.assert_allclose(output["probabilities"].sum(0), 1, atol=2e-6)

    def test_inference_requires_frozen_eval_and_rejects_nonfinite_logits(self):
        gray = np.zeros((10, 10), np.uint8)
        with self.assertRaises(ValueError):
            predict(FixedLogits(), gray, "cpu", 32)
        model = FixedLogits().eval()
        model.register_parameter("parameter", nn.Parameter(torch.ones(1)))
        with self.assertRaises(ValueError):
            predict(model, gray, "cpu", 32)
        model.requires_grad_(False)
        model.logits[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            predict(model, gray, "cpu", 32)

    def test_convmodule_matches_native_sync_batchnorm_eval(self):
        ours = ConvModule(4, 7).eval()
        reference = nn.Sequential(nn.Conv2d(4, 7, 1, bias=False), nn.SyncBatchNorm(7), nn.ReLU(inplace=True)).eval()
        reference[0].load_state_dict(ours.conv.state_dict())
        with torch.no_grad():
            ours.bn.running_mean.copy_(torch.arange(7).float()/3)
            ours.bn.running_var.copy_(torch.arange(7).float()/4+0.5)
        reference[1].load_state_dict(ours.bn.state_dict())
        values = torch.randn(2, 4, 11, 17)
        torch.testing.assert_close(ours(values), reference(values), rtol=0, atol=0)

    def test_checkpoint_rejects_encoder_only_wrong_classes_and_wrong_depth(self):
        state = {"encoder.patch_embed1.proj.weight": torch.zeros(64, 3, 7, 7),
                 "decoder.linear_c1.proj.weight": torch.zeros(256, 64),
                 "decoder.linear_pred.weight": torch.zeros(15, 256, 1, 1),
                 "decoder.linear_pred.bias": torch.zeros(15), "classifier.weight": torch.zeros(15, 512, 1, 1)}
        with self.assertRaisesRegex(ValueError, "complete FMB"):
            validate_state_dict({"encoder.patch_embed1.proj.weight": state["encoder.patch_embed1.proj.weight"]})
        for stage, depth in enumerate((3, 4, 18, 3), 1):
            for i in range(depth):
                state[f"encoder.block{stage}.{i}.norm1.weight"] = torch.ones(1)
        validate_state_dict(state)
        validate_state_dict({"module." + key: value for key, value in state.items()})
        wrong = dict(state, **{"decoder.linear_pred.bias": torch.zeros(14)})
        with self.assertRaises(ValueError):
            validate_state_dict(wrong)
        del state["encoder.block3.17.norm1.weight"]
        with self.assertRaisesRegex(ValueError, "depth"):
            validate_state_dict(state)

    def test_data_state_rejects_non_tensor_keys(self):
        with self.assertRaises(ValueError):
            validate_state_dict({1: torch.ones(1)})

    def fixture_review(self, root):
        image_dir = root / "trainB"
        image_dir.mkdir()
        source = image_dir / "fixed.png"
        Image.fromarray(np.zeros((288, 360), np.uint8)).save(source)
        protocol = root / "manifest.csv"
        with protocol.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["destination_name", "destination_split", "video_id"])
            writer.writeheader(); writer.writerow(dict(destination_name="fixed.png", destination_split="trainB", video_id="video"))
        review = dict(geometry="full", selection=dict(seed=42, samples=[dict(filename="fixed.png",video_id="video",group="regular")]),
                      source_hashes={"fixed.png":sha256_file(source)},
                      items=[dict(id="01_O01",filename="fixed.png",category="person",group="regular",priority=True,
                                  crop=[11,12,31,42],box=[12.,13.,30.,41.])])
        manifest=root/"review.json";manifest.write_text(json.dumps(review))
        return SimpleNamespace(review_manifest=manifest,protocol_manifest=protocol,input_dir=image_dir,num_images=1),review

    def test_review_preflight_checks_image_hash_train_split_and_crop_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            args, review = self.fixture_review(Path(directory))
            _, selected, targets = preview_semantics.load_review(args)
            self.assertEqual((len(selected),len(targets)),(1,1))
            Image.fromarray(np.ones((288,360),np.uint8)).save(args.input_dir/"fixed.png")
            with self.assertRaisesRegex(ValueError,"changed"):
                preview_semantics.load_review(args)
            review["source_hashes"]["fixed.png"]=sha256_file(args.input_dir/"fixed.png")
            review["items"][0]["crop"][2]=361
            args.review_manifest.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,"crop geometry"):
                preview_semantics.load_review(args)
            review["items"][0]["crop"][2]=31
            args.review_manifest.write_text(json.dumps(review))
            args.protocol_manifest.write_text("destination_name,destination_split,video_id\nfixed.png,testB,video\n")
            with self.assertRaisesRegex(ValueError,"trainB protocol"):
                preview_semantics.load_review(args)

    def test_review_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            args, review = self.fixture_review(Path(directory))
            review["selection"]["samples"][0]["filename"]="../fixed.png"
            args.review_manifest.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,"basenames"):
                preview_semantics.load_review(args)

    def test_render_crop_uses_full_frame_results_and_fixed_score_range(self):
        gray = np.zeros((20,30),np.uint8)
        result = predict(FixedLogits().eval(),gray,"cpu",32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"crop.png"
            render_crop(gray,{"base":result},[2,3,7,11],path,4)
            with Image.open(path) as image:
                self.assertEqual(image.size,(1050,60))
            maps=score_maps(result)
            np.testing.assert_allclose(maps["vehicles"],result["probabilities"][9:14].sum(0),rtol=0,atol=0)

    def test_html_is_offline_and_preserves_javascript_newline_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"index.html"
            write_page(path,[],[],["base"],"pack",{"short_side":512,"detail_short_side":0})
            page=path.read_text()
            self.assertIn("lines.join('\\n')",page)
            self.assertNotIn("lines.join('\n')",page)
            self.assertNotIn('<script src=',page)
            self.assertIn("semantic_review_v1",page)

    def test_cli_protects_author_baseline_and_existing_output(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                preview_semantics.parse_args(["--short-side","768"])
            with self.assertRaises(SystemExit):
                preview_semantics.parse_args(["--detail-short-side","0","--flip-tta"])
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit):
                    preview_semantics.parse_args(["--output-dir",directory])


if __name__ == "__main__":
    unittest.main()
