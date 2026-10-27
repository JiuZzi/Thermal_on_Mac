import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from D.common import load_config, control_tensor, select_sources, read_condition, sha256, write_json
from D.preview import initialize_output


class ProtocolTests(unittest.TestCase):
    def test_soft_strength_survives_channel_replication_and_resize(self):
        edge = np.full((8, 10), .37, np.float32)
        actual = control_tensor(edge, (20, 16)).numpy()
        self.assertEqual(actual.shape, (1, 3, 16, 20))
        np.testing.assert_allclose(actual, .37, atol=1e-7)
        for bad in (np.array([[np.nan]]), np.array([[1.01]])):
            with self.assertRaises(ValueError):
                control_tensor(bad, (20, 16))

    def test_video_split_is_preserved_and_sampling_covers_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rows = []
            for scene in range(3):
                for index in range(4):
                    name = f"s{scene}_{index}.png"
                    Image.new("L", (10, 8), index).save(base / name)
                    rows.append(dict(image=name, group=f"scene{scene}", split="val"))
            Image.new("RGB", (10, 8)).save(base / "a.png")
            data = dict(format="grouped_unaligned_v1", domains=dict(
                A=[dict(image="a.png", group="other", split="train")], B=rows))
            manifest = base / "manifest.json"; write_json(manifest, data)
            one, _ = select_sources(manifest, limit=3)
            two, _ = select_sources(manifest, limit=3)
            self.assertEqual(one, two)
            self.assertEqual(len({row["group"] for row in one}), 3)
            data["domains"]["A"][0]["group"] = "scene0"
            write_json(manifest, data)
            with self.assertRaisesRegex(ValueError, "leaks"):
                select_sources(manifest, limit=3)

    def test_cache_mutation_and_geometry_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "one.npz"
            np.savez(path, TIR_uint8=np.zeros((8, 10), np.uint8), E=np.zeros((8, 10), np.float32))
            frame = dict(condition="one.npz", condition_sha256=sha256(path), native_size=[10, 8])
            self.assertEqual(read_condition(base, frame)[0].size, (10, 8))
            frame["native_size"] = [8, 10]
            with self.assertRaises(ValueError):
                read_condition(base, frame)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                read_condition(base, frame)

    def test_resume_does_not_mix_conditions_or_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            identity = dict(variants=["D0", "D1"], config=load_config(), conditions_sha256="one")
            initialize_output(output, identity, False)
            initialize_output(output, identity, True)
            with self.assertRaisesRegex(ValueError, "Preserve"):
                initialize_output(output, identity, False)
            changed = copy.deepcopy(identity); changed["config"]["strength"] = .8
            with self.assertRaisesRegex(ValueError, "identical"):
                initialize_output(output, changed, True)

    def test_config_is_pinned_and_rejects_invalid_denoising_settings(self):
        config = load_config()
        self.assertEqual(config["seed"], 42)
        self.assertEqual(len(config["base_revision"]), 40)
        with tempfile.TemporaryDirectory() as directory:
            config["strength"] = .001
            path = Path(directory) / "config.json"; write_json(path, config)
            with self.assertRaisesRegex(ValueError, "denoising"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
