"""Geometry and split checks for the auxiliary FLIR ROI detector."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from train_flir_roi_detector import processed_boxes, select_records
from models.thermal_roi_detector import boxes_to_importance
from models.saliency_edges import object_line_contour


class ThermalRoiDetectorTests(unittest.TestCase):
    def test_source_boxes_follow_offline_flir_geometry(self):
        record = {
            "source_width": 640, "source_height": 512,
            "annotations": [{"bbox": [160, 128, 128, 128], "category_id": 3}],
        }
        boxes, labels = processed_boxes(record)
        np.testing.assert_allclose(boxes.numpy(), [[55, 44, 155, 144]], atol=1e-6)
        self.assertEqual(labels.tolist(), [3])

    def test_video_holdout_never_enters_training(self):
        records = [
            {"filename": "a.jpg", "video_id": "train"},
            {"filename": "b.jpg", "video_id": "train"},
            {"filename": "c.jpg", "video_id": "validation"},
        ]
        train, val = select_records(records, {"validation"}, 42, None, None)
        self.assertEqual({row["filename"] for row in train}, {"a.jpg", "b.jpg"})
        self.assertEqual({row["filename"] for row in val}, {"c.jpg"})

    def test_soft_box_roi_retains_boundary_shoulder(self):
        importance = boxes_to_importance((32, 32), [((10, 10, 20, 20), 0.9, 3)])
        self.assertEqual(importance.shape, (32, 32))
        self.assertAlmostEqual(float(importance[15, 15]), 1.0, places=6)
        self.assertGreater(float(importance[15, 21]), 0.1)
        self.assertLess(float(importance[0, 0]), 0.2)
        empty = boxes_to_importance((32, 32), [])
        np.testing.assert_allclose(empty, 0.1)

    def test_long_line_region_only_changes_second_channel_candidate(self):
        gray = np.zeros((32, 32), dtype=np.float32)
        gray[:, 16:] = 1.0
        object_roi = np.full((32, 32), 0.1, dtype=np.float32)
        no_lines = object_line_contour(gray, object_roi, np.zeros_like(gray), 0.5)
        line_region = object_line_contour(gray, object_roi, np.ones_like(gray), 0.5)
        self.assertEqual(no_lines.shape, gray.shape)
        self.assertTrue(np.isfinite(line_region).all())
        self.assertTrue(np.all((line_region >= 0) & (line_region <= 1)))
        self.assertGreater(float((line_region - no_lines).max()), 0)


if __name__ == "__main__":
    unittest.main()
