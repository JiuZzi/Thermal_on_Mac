"""Regression tests for test-only geometry, empty targets and registered pairs."""

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

import numpy as np

from D.common import write_json
from D.unpaired.boundary_metrics import matched_counts, scores
from D.unpaired.paired_metrics import FORMAT, aligned_pairs
from D.unpaired.test_metrics import coco_scores, flir_ground_truth, processed_bbox


class TestOnlyMetrics(unittest.TestCase):
    def test_flir_box_resize_crop_and_empty(self):
        self.assertEqual(processed_bbox([0, 0, 160, 160], (640, 512)), [0, 0, 55, 69])
        self.assertIsNone(processed_bbox([0, 0, 16, 16], (640, 512)))

    @unittest.skipUnless(find_spec("pycocotools"), "optional COCO evaluator is not installed")
    def test_coco_categorizes_video_without_gt_and_empty_detections(self):
        coco = dict(
            images=[dict(id=index, file_name=f"{name}.jpg", width=640, height=512)
                    for index, name in enumerate(("one", "two"), 1)],
            categories=[dict(id=10, name="person"), dict(id=12, name="car")],
            annotations=[dict(image_id=1, category_id=10, bbox=[100, 100, 32, 64], iscrowd=0)],
        )
        sources = [dict(filename=f"{name}.jpg", group=name) for name in ("one", "two")]
        truth = flir_ground_truth(coco, sources)
        box = truth["annotations"][0]
        prediction = [dict(image_id=1, category_id=1, bbox=box["bbox"], score=0.9)]
        report = coco_scores(truth, prediction)
        self.assertAlmostEqual(report["overall"]["ap"], 1)
        self.assertAlmostEqual(report["overall"]["ap75"], 1)
        self.assertIsNone(report["overall"]["classes"]["car"]["ap"])
        self.assertIsNone(report["by_video"]["two"]["ap"])
        self.assertEqual(coco_scores(truth, [])["overall"]["ap"], 0)

    def test_boundary_matching_counts_each_pixel_at_most_once(self):
        reference = np.zeros((12, 12), dtype=bool)
        reference[5, 5] = True
        predicted = reference.copy()
        predicted[5, 7] = True
        predicted[5, 8] = True
        roi = np.ones_like(reference)
        roi[5, 8] = False
        values = scores(matched_counts(predicted, reference, roi))
        self.assertEqual(values["precision"], 0.5)
        self.assertEqual(values["recall"], 1)
        self.assertAlmostEqual(values["f1"], 2 / 3)
        self.assertEqual(scores(matched_counts(np.zeros_like(predicted), reference, roi))["recall"], 0)

    def test_paired_input_rejects_unmatched_stems(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "pairs.json"
            payload = dict(format=FORMAT, dataset="fixture", pairs=[dict(tir="tir/a.png", rgb="rgb/a.png")])
            write_json(path, payload)
            _, _, rows = aligned_pairs(path, root / "generated")
            self.assertEqual(rows["a"][2].name, "a_fake_B.png")
            payload["pairs"][0]["rgb"] = "rgb/b.png"
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "shared pair stem"):
                aligned_pairs(path, root / "generated")


if __name__ == "__main__":
    unittest.main()
