"""Train-only appearance reference, population gradients and color details."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from D.common import ROOT, sha256, write_json
from D.unpaired.appearance import (CLASSES, RegionAppearanceDistribution, fit_reference, lab_cdf,
                                   load_reference, region_samples)
from D.unpaired.appearance_metrics import interior_chroma_jump
from D.unpaired.quality import rgb_appearance
from D.unpaired.train import training_config


CONFIG = ROOT / "D/unpaired/config_quality_v3_appearance.json"


def region(rgb_color, name="person", size=(32, 48)):
    rgb = torch.tensor(rgb_color, dtype=torch.float32)[None, :, None, None].expand(1, 3, *size).clone()
    probabilities = torch.zeros(1, len(CLASSES) + 1, *size)
    probabilities[:, CLASSES.index(name) + 1] = 1
    return rgb, probabilities, torch.ones(1, 1, *size)


class AppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_opt_in_config_does_not_change_forward_settings(self):
        base = training_config(ROOT / "D/unpaired/config_quality_v3.json")
        enabled = training_config(CONFIG)
        self.assertEqual(enabled["quality"], base["quality"])
        self.assertEqual(enabled["loss_weights"]["edge"], base["loss_weights"]["edge"])
        self.assertEqual(enabled["loss_weights"]["content"], base["loss_weights"]["content"])
        self.assertEqual(enabled["loss_weights"]["color_prior"], 0)
        self.assertEqual(enabled["loss_weights"]["color_attraction"], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            enabled["loss_weights"]["cycle_rgb"] = 0
            write_json(path, enabled)
            with self.assertRaisesRegex(ValueError, "appearance and RGB cycle"):
                training_config(path)

    def test_fit_is_group_balanced_and_train_only(self):
        colors = [("train", "long", (255, 64, 64))] * 3 + [("train", "short", (32, 64, 255)),
                                                            ("val", "heldout", (64, 255, 32))]
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            frames = []
            expected = []
            for index, (role, group, color) in enumerate(colors):
                rgb, p, q = region([2 * channel / 255 - 1 for channel in color])
                if index in (0, 3):
                    lab, _ = region_samples(rgb, p, q, .5)
                    expected.append(lab_cdf(lab)[0, CLASSES.index("person")])
                file = cache / f"{index}.npz"
                image = np.broadcast_to(np.asarray(color, dtype=np.uint8), (*rgb.shape[-2:], 3))
                gray = np.zeros(rgb.shape[-2:], dtype=np.uint8)
                np.savez_compressed(file, RGB=image, gray=gray, E=gray.astype(np.float32),
                                    P=p[0].numpy(), Q=q[0, 0].numpy())
                frames.append(dict(id=f"id{index}", stem=f"frame{index}", domain="RGB", role=role, group=group,
                                   condition=file.name, condition_sha256=sha256(file)))
            manifest = cache / "manifest.json"
            write_json(manifest, dict(format="D_unpaired_conditions_v1", identity=dict(dataset_id="other_sensor"),
                                      frames=frames))
            write_json(cache / "status.json", dict(phase="completed", total=len(frames)))
            ref = fit_reference(cache, .5)
            path = cache / "reference.json"
            write_json(path, ref)
            target, supported, digest = load_reference(path, sha256(manifest), .5)
            index = CLASSES.index("person")
            torch.testing.assert_close(target[index], torch.stack(expected).mean(dim=0))
            self.assertEqual(ref["sampled_frames"], 4)
            self.assertEqual(ref["group_counts"][index], 2)
            self.assertEqual(supported.sum().item(), 1)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "exact RGB training cache"):
                load_reference(path, "wrong_digest", .5)
            write_json(path, dict(ref, role="val"))
            with self.assertRaisesRegex(ValueError, "exact RGB training cache"):
                load_reference(path, sha256(manifest), .5)

    def test_distribution_prefers_color_modes_over_gray_collapse(self):
        red, p, q = region((.7, -.45, -.4), "car")
        blue, _, _ = region((-.45, -.4, .7), "car")
        gray, _, _ = region((0., 0., 0.), "car")
        lab_red, active = region_samples(red, p, q, .5)
        lab_blue, _ = region_samples(blue, p, q, .5)
        supported = active[0]
        cdf = ((lab_cdf(lab_red) + lab_cdf(lab_blue)) / 2)[0] * supported[:, None, None]
        reference = (cdf, supported, "fixture")
        varied = RegionAppearanceDistribution(reference, .5, "cpu")
        collapsed = RegionAppearanceDistribution(reference, .5, "cpu")
        for step in range(48):
            x = red if step % 2 == 0 else blue
            varied_loss, varied.history, _ = varied.loss(x, p, q, step=step)
            gray_loss, collapsed.history, _ = collapsed.loss(gray, p, q, step=step)
        self.assertGreater(float(gray_loss), 10 * float(varied_loss))
        gray.requires_grad_()
        zero, next_history, parts = collapsed.loss(gray, p, q * 0, step=48)
        self.assertEqual(float(zero), 0)
        self.assertEqual(float(parts["appearance_dist_classes"]), 0)
        torch.testing.assert_close(next_history, collapsed.history)
        loss, _, _ = collapsed.loss(gray, p, q, step=48)
        grad, = torch.autograd.grad(loss, gray)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_chroma_detail_sees_an_unsupported_seam(self):
        rgb, p, q = region((0., 0., 0.))
        changed = rgb.clone()
        changed[:, 0, :, :24] += .4
        changed[:, 1, :, :24] -= .4 * .299 / .587
        changed.requires_grad_()
        settings = training_config(CONFIG)["quality"]
        loss, diagnostics = rgb_appearance(changed, rgb, settings, chroma_detail=.5)
        self.assertGreater(float(diagnostics["rgb_appearance_chroma_detail"]), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        _, same = rgb_appearance(rgb, rgb, settings, chroma_detail=.5)
        self.assertEqual(float(same["rgb_appearance_chroma_detail"]), 0)
        condition = torch.zeros_like(rgb)
        sample = dict(x=condition, p=p, q=q)
        jump = interior_chroma_jump(changed.detach(), sample, .5)
        condition[:, 1, :, 23:25] = 1
        supported_jump = interior_chroma_jump(changed.detach(), sample, .5)
        index = CLASSES.index("person")
        self.assertGreater(float(jump[0, index]), 0)
        self.assertEqual(float(supported_jump[0, index]), 0)
