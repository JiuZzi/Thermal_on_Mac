"""Audit geometry, blind sampling and point-score contracts, not model quality."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_rgb_semantics import (MEAN, STD, POINTS, predict_visible, select_records,
                                summarize_maps, point_metrics, read_annotations)
from semantic_preview.inference import predict_probabilities


class InputLogits(nn.Module):
    def forward(self, image):
        self.seen = image.detach().clone()
        values = F.adaptive_avg_pool2d(image[:, :1] - image[:, 2:], (3, 5))
        return torch.cat([values * (i - 7) / 10 for i in range(15)], 1)


class RGBAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gray_probe_exactly_reuses_unchanged_thermal_preprocess_on_visible_luminance(self):
        rgb = np.random.default_rng(42).integers(0, 256, (19, 27, 3), dtype=np.uint8)
        model = InputLogits().eval().requires_grad_(False)
        actual = predict_visible(model, rgb, "gray", short_side=32)
        expected = predict_probabilities(model, np.asarray(Image.fromarray(rgb).convert("L")), "cpu", short_side=32)
        np.testing.assert_array_equal(actual, expected)

    def test_rgb_probe_retains_channel_order_and_restores_logits_before_softmax(self):
        rgb = np.zeros((19, 27, 3), np.uint8)
        rgb[:, :, 0] = np.arange(27, dtype=np.uint8)[None] * 8
        rgb[:, :, 1] = 90
        rgb[:, :, 2] = np.arange(19, dtype=np.uint8)[:, None] * 10
        model = InputLogits().eval().requires_grad_(False)
        actual = predict_visible(model, rgb, "rgb", short_side=32)
        resized = np.asarray(Image.fromarray(rgb).resize((int(32/19*27), 32), Image.Resampling.BILINEAR))
        expected_tensor = torch.from_numpy(((resized.astype(np.float32)-MEAN)/STD).transpose(2, 0, 1).copy())[None]
        torch.testing.assert_close(model.seen, expected_tensor, rtol=0, atol=0)
        expected = F.interpolate(model(expected_tensor), size=(19, 27), mode="bilinear", align_corners=True).softmax(1)[0]
        np.testing.assert_array_equal(actual, expected.numpy())
        gray = predict_visible(model, rgb, "gray", short_side=32)
        self.assertGreater(float(np.abs(actual-gray).max()), .01)

    def test_modes_require_eval_freezing_and_valid_rgb(self):
        rgb = np.zeros((12, 15, 3), np.uint8)
        for mode in ("rgb", "gray"):
            with self.assertRaises(ValueError):
                predict_visible(InputLogits(), rgb, mode, short_side=32)
            model = InputLogits().eval()
            model.register_parameter("trainable", nn.Parameter(torch.ones(1)))
            with self.assertRaises(ValueError):
                predict_visible(model, rgb, mode, short_side=32)
        for image in (rgb.astype(np.uint16), rgb[:, :, 0], np.zeros((0, 3, 3), np.uint8)):
            with self.assertRaises(ValueError):
                predict_visible(InputLogits().eval(), image, "rgb", short_side=32)

    def test_sampling_is_deterministic_distinct_video_and_train_only(self):
        records = [dict(destination_split="trainA", video_id=str(i//3), scene=str(i//9), destination_name=f"{i}.jpg") for i in range(36)]
        records.append(dict(destination_split="testA", video_id="forbidden", scene="new", destination_name="test.jpg"))
        first = select_records(records, count=9)
        self.assertEqual(first, select_records(records, count=9))
        self.assertEqual(len({r["video_id"] for r in first}), 9)
        self.assertEqual({r["scene"] for r in first}, {"0", "1", "2", "3"})
        self.assertTrue(all(r["destination_split"] == "trainA" for r in first))
        with self.assertRaises(ValueError):
            select_records(records, count=13)

    def test_candidate_cores_reject_ambiguous_scores_and_erode_boundaries(self):
        p = np.zeros((15, 15, 17), np.float32)
        p[6], p[7] = .95, .05
        labels, maximum, margin, core = summarize_maps(p)
        self.assertEqual(core[7, 8], 6)
        self.assertEqual(core[0, 0], 255)
        p[6], p[7] = .55, .45
        self.assertTrue((summarize_maps(p)[3] == 255).all())

    def test_point_scores_count_wrong_high_score_cores_and_exclude_uncertain(self):
        points = [dict(expected="vegetation", predicted="sky", core_group="sky"),
                  dict(expected="sky", predicted="sky", core_group="sky"),
                  dict(expected="other", predicted="vegetation", core_group=None),
                  dict(expected="uncertain", predicted="sky", core_group="sky")]
        result = point_metrics(points)
        self.assertEqual(result["evaluated_points"], 3)
        self.assertEqual(result["candidate_core_point_precision"], .5)
        self.assertEqual(result["per_group"]["sky"]["precision"], .5)
        self.assertEqual(result["per_group"]["vegetation"]["recall"], 0)

    def test_annotation_reader_rejects_unfinished_labels_and_moved_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = dict(points=[dict(id=f"01_P{i}",prefix="01",x=x,y=y,expected="road")
                                for i,(x,y) in enumerate(POINTS,1)])
            path = root / "manual_points.json"
            path.write_text(json.dumps(data))
            manifest = dict(frames=[dict(prefix="01")])
            read_annotations(root,manifest)
            data["points"][0]["expected"] = None
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_annotations(root,manifest)
            data["points"][0]["expected"] = "road"
            data["points"][0]["x"] += 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_annotations(root,manifest)


if __name__ == "__main__":
    unittest.main()
