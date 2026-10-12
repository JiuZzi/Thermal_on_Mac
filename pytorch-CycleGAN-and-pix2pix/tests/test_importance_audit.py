"""CPU regression gates for the isolated, diagnostic-only importance ablation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_c2_importance import fuse_fixed_candidates, importance_ablation, map_stats, priority_objects


class ImportanceAuditTests(unittest.TestCase):
    def test_replica_matches_current_c2c_and_only_R_changes(self):
        image = np.zeros((64, 64), np.float32)
        image[12:52, 16:48] = .8
        image[30:32, :] = .4
        maps = importance_ablation(image)
        mean = np.float32(maps["R"].mean(dtype=np.float64))
        np.testing.assert_array_equal(maps["constant_R"], np.full_like(image, mean))
        expected, _ = fuse_fixed_candidates(maps["fine"], maps["coarse"], maps["constant_R"], maps["clutter"])
        np.testing.assert_array_equal(expected, maps["constant"])
        for tag in ("spatial", "constant"):
            self.assertTrue(np.all(maps[tag] >= .25 * maps["fine"]))
            self.assertTrue(np.all((maps[tag] >= 0) & (maps[tag] <= 1)))

    def test_constant_importance_makes_both_controls_identical(self):
        fine = np.full((8, 8), .8, np.float32)
        coarse = np.full_like(fine, .4)
        importance = np.full_like(fine, .32)
        clutter = np.full_like(fine, .5)
        spatial, _ = fuse_fixed_candidates(fine, coarse, importance, clutter)
        constant, _ = fuse_fixed_candidates(fine, coarse, np.full_like(fine, importance.mean()), clutter)
        np.testing.assert_allclose(spatial, constant, atol=1e-7, rtol=0)

    def test_fixed_fusion_does_not_mutate_inputs(self):
        values = [np.full((8, 8), value, np.float32) for value in (.8, .4, .32, .5)]
        originals = [value.copy() for value in values]
        fuse_fixed_candidates(*values)
        for value, original in zip(values, originals):
            np.testing.assert_array_equal(value, original)

    def test_empty_candidates_stay_empty(self):
        maps = importance_ablation(np.zeros((32, 32), np.float32))
        for tag in ("fine", "coarse", "spatial", "constant"):
            self.assertEqual(np.count_nonzero(maps[tag]), 0)
        self.assertIsNone(map_stats(maps)["spatial_floor_active_on_candidates"])

    def test_floor_can_mask_R_change(self):
        fine = np.ones((4, 4), np.float32)
        zero = np.zeros_like(fine)
        a, raw_a = fuse_fixed_candidates(fine, zero, np.full_like(fine, .18), zero)
        b, raw_b = fuse_fixed_candidates(fine, zero, np.full_like(fine, .2), zero)
        self.assertTrue(np.any(raw_a != raw_b))
        np.testing.assert_array_equal(a, b)

    def test_invalid_shapes_and_values_are_rejected(self):
        valid = np.zeros((4, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            fuse_fixed_candidates(valid, valid, np.zeros((2, 2)), valid)
        for value in (np.nan, -.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                fuse_fixed_candidates(valid, valid, np.full_like(valid, value), valid)

    def test_roi_keeps_full_frame_constant_R(self):
        image = np.zeros((64, 64), np.float32)
        image[8:40, 8:40] = .8
        maps = importance_ablation(image)
        crop = {key: value[10:30, 10:30] for key, value in maps.items()}
        np.testing.assert_array_equal(crop["constant_R"], np.full((20, 20), maps["constant_R"][0, 0], np.float32))
        # A crop is not allowed to silently recompute its own mean importance.
        rebuilt, _ = fuse_fixed_candidates(crop["fine"], crop["coarse"], crop["constant_R"], crop["clutter"])
        np.testing.assert_array_equal(crop["constant"], rebuilt)

    def test_priority_selection_uses_boxes_not_map_values(self):
        record = dict(source_width=500, source_height=400, annotations=[
            dict(id=1, category_id=1, bbox=[100, 100, 50, 100]),
            dict(id=2, category_id=1, bbox=[200, 100, 20, 40]),
            dict(id=3, category_id=3, bbox=[110, 200, 20, 20]),
        ])
        selected = priority_objects(record, "full")
        self.assertEqual([item["annotation_id"] for item in selected], [2, 3, 1])


if __name__ == "__main__":
    unittest.main()
