from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from D.common import sha256, write_json
from D.prepare_rgb import CITY_TO_FMB, NAMES, map_probabilities, resume_cache


class PrepareRGBTests(unittest.TestCase):
    def test_group_sum_roundoff_is_repaired_and_strict_contour_validation_accepts_it(self):
        names = list(CITY_TO_FMB)
        p = np.zeros((19, 3, 4), np.float32)
        p[names.index("wall")] = .5
        p[names.index("fence")] = .5
        p[names.index("terrain")] = np.finfo(np.float32).eps
        mapped = map_probabilities(p, names)
        self.assertEqual(mapped.dtype, np.float32)
        self.assertTrue(np.isfinite(mapped).all())
        self.assertGreaterEqual(float(mapped.min()), 0.)
        self.assertLessEqual(float(mapped.max()), 1.)
        np.testing.assert_array_equal(mapped[NAMES.index("background")], 1.)
        import sys
        from D.common import ROOT
        sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
        from models.semantic_edges import semantic_importance
        importance, _ = semantic_importance(mapped)
        np.testing.assert_allclose(importance, .18)

    def test_invalid_probabilities_still_fail_instead_of_being_hidden(self):
        names = list(CITY_TO_FMB)
        valid = np.full((19, 2, 3), 1 / 19, np.float32)
        for value in (np.nan, np.inf, -.01, 1.01):
            p = valid.copy(); p[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid source"):
                map_probabilities(p, names)
        with self.assertRaises(ValueError):
            map_probabilities(valid * .5, names)

    def fixture(self, base):
        output = base / "cache"; (output / "maps").mkdir(parents=True)
        rgb = np.full((3, 4, 3), 127, np.uint8)
        for name in ("first.png", "second.png"):
            Image.fromarray(rgb).save(base / name)
        path = output / "maps/train_first.npz"
        np.savez_compressed(path, RGB_uint8=rgb, E=np.full((3, 4), .4, np.float32))
        rows = [dict(image="first.png", split="train", group="scene1"),
                dict(image="second.png", split="val", group="scene2")]
        expected = dict(format="thermal_D_rgb_selfpairs_v1", seed=42, settings={"x":(1, 2)}, frames=[])
        frame = dict(stem="first", filename="first.png", group="scene1", role="train", source="first.png",
                     source_sha256=sha256(base / "first.png"), native_size=[4, 3],
                     condition="maps/train_first.npz", condition_sha256=sha256(path))
        write_json(output / "manifest.json", dict(expected, frames=[frame]))
        write_json(output / "status.json", dict(phase="preparing", completed=1, total=2))
        return output, expected, rows, path

    def test_resume_accepts_legacy_prefix_without_rewriting_and_revalidates_new_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); output, expected, rows, path = self.fixture(base)
            original = path.read_bytes()
            saved, seen = resume_cache(output, expected, rows, base)
            self.assertEqual(len(saved["frames"]), 1)
            self.assertEqual(saved["probability_fix"]["first_frame"], 2)
            self.assertEqual(list(seen.values()), ["train"])
            self.assertEqual(path.read_bytes(), original)
            write_json(output / "manifest.json", saved)
            again, _ = resume_cache(output, expected, rows, base)
            self.assertEqual(again, saved)

    def test_resume_rejects_changed_source_settings_or_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); output, expected, rows, path = self.fixture(base)
            with self.assertRaisesRegex(ValueError, "identical"):
                resume_cache(output, dict(expected, seed=43), rows, base)
            path.write_bytes(b"changed cache")
            with self.assertRaisesRegex(ValueError, "changed"):
                resume_cache(output, expected, rows, base)


if __name__ == "__main__":
    unittest.main()
