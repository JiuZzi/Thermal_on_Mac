"""Contour geometry, online dispatch, checkpoint identity and cache replay gates."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix/tests"))

from models.selective_semantic_edges import importance_gain, consolidate_response, selective_components
from models.semantic_edges import FMB_CLASS_NAMES, SETTINGS, IMPORTANT_CLASSES, make_structure_semantic_condition
from models.semantic_contour import SemanticContourGenerator
from models.semantic_color import SemanticColorGenerator
from models.learned_color import LearnedColorGenerator
from models import networks
from test_semantic_contour_training import FakePrior, fixture
from test_learned_color_training import bank_file
from preview_semantic_contours import preflight
from semantic_preview.checkpoint import sha256_file


def scores(name, shape):
    p = np.zeros((15, *shape), np.float32)
    p[FMB_CLASS_NAMES.index(name)] = 1
    return p


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gain_attenuates_background_without_amplifying_foreground(self):
        r = np.array([[.18, .515, .85]], np.float32)
        np.testing.assert_allclose(importance_gain(r), [[.25, .625, 1]], atol=1e-6)
        for invalid in (np.array([[0]], np.float32), r + np.nan):
            with self.assertRaises(ValueError): importance_gain(invalid)

    def test_broad_response_narrows_and_keeps_its_center_peak(self):
        x = np.arange(64, dtype=np.float32)
        gray = np.tile(x / 63, (48, 1))
        response = np.tile(np.exp(-.5 * ((x - 32) / 3) ** 2), (48, 1)).astype(np.float32)
        result, ridge = consolidate_response(gray, response)
        self.assertEqual(np.flatnonzero(ridge[24]).tolist(), [32])
        self.assertEqual(result[24, 32], response[24, 32])
        self.assertLess(result.sum(), .5 * response.sum())
        self.assertTrue(np.all(result <= response))

    def test_two_boundaries_stay_separate_and_no_gap_is_completed(self):
        gray = np.tile(np.arange(64, dtype=np.float32) / 63, (48, 1))
        response = np.zeros_like(gray)
        response[:, 20] = 1; response[:, 40] = .8
        response[20:24, 20] = 0
        result, _ = consolidate_response(gray, response)
        np.testing.assert_array_equal(result[response == 0], 0)
        self.assertGreater(result[10, 20], .9)
        self.assertGreater(result[10, 40], .7)

    def test_flat_tir_never_turns_semantic_regions_into_edges(self):
        gray = np.full((48, 64), .5, np.float32)
        p = scores("sky", gray.shape)
        p[:, 10:30, 20:35] = scores("person", (20, 15))
        maps = selective_components(gray, p)
        for key in ("joint512", "gain_only", "ridge_only", "selective_v1"):
            np.testing.assert_array_equal(maps[key], 0)

    def test_old_control_is_exact_and_new_maps_only_select_existing_responses(self):
        rng = np.random.default_rng(917)
        for shape in ((32, 40), (64, 64)):
            gray = rng.random(shape, dtype=np.float32)
            p = rng.random((15, *shape), dtype=np.float32); p /= p.sum(0, keepdims=True)
            for image, prob in ((gray, p), (gray[:, ::-1], p[:, :, ::-1])):
                maps = selective_components(image, prob)
                np.testing.assert_array_equal(maps["joint512"], make_structure_semantic_condition(image, prob))
                for key in ("gain_only", "ridge_only", "selective_v1"):
                    self.assertEqual(maps[key].dtype, np.float32)
                    self.assertTrue(np.isfinite(maps[key]).all())
                    self.assertTrue(np.all((maps[key] >= 0) & (maps[key] <= maps["joint512"])))
                    self.assertTrue(np.all(maps[key] >= SETTINGS["fine_floor"] * maps["fine"]))

    def test_wrong_background_semantics_cannot_remove_the_original_fine_floor(self):
        gray = (fixture()[0, 0].numpy() + 1) / 2
        maps = selective_components(gray, scores("sky", gray.shape))
        self.assertGreater(maps["fine"].sum(), 0)
        self.assertTrue(np.all(maps["selective_v1"] >= .25 * maps["fine"]))

    def test_channel_contract_and_online_candidates_match_preview(self):
        with patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            generator = SemanticContourGenerator(nn.Identity(), "stub", contour_variant="selective_v1")
        tir = fixture()
        actual = generator(tir)
        gray = (tir[0, 0].numpy() + 1) / 2
        p = generator.semantic_prior.predict(gray, "cpu")
        np.testing.assert_array_equal(actual[0, 1].numpy(), selective_components(gray, p)["selective_v1"])
        torch.testing.assert_close(actual[:, :1], tir, atol=0, rtol=0)
        self.assertEqual(actual[:, 2:].count_nonzero(), 0)

    def test_semantic_and_learned_color_paths_honor_selected_contour(self):
        with tempfile.TemporaryDirectory() as tmp, patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            backbone = networks.ResnetGenerator(3, 3, ngf=8, n_blocks=1,
                                               norm_layer=networks.get_norm_layer("instance"))
            semantic = SemanticColorGenerator(backbone, "stub", color_condition="semantic", color_width=4,
                                              contour_variant="selective_v1")
            learned = LearnedColorGenerator(semantic, bank_file(Path(tmp)), width=4, dropout=0.)
            tir = fixture(); gray = (tir[0, 0].numpy() + 1) / 2
            p = torch.from_numpy(semantic.semantic_prior.predict(gray, "cpu"))[None]
            _, sem_aux = semantic(tir, return_aux=True, probabilities_override=p)
            _, aux = learned(tir, return_aux=True, probabilities_override=p)
            expected = selective_components(gray, p[0].numpy())["selective_v1"]
            np.testing.assert_array_equal(aux["edge"][0, 0].numpy(), expected)
            torch.testing.assert_close(aux["edge"], sem_aux["edge"], atol=0, rtol=0)
            learned.feature_override = "off"
            _, off = learned(tir, return_aux=True, probabilities_override=p)
            self.assertEqual(off["feature_gate"].count_nonzero(), 0)

    def test_legacy_signature_is_unchanged_and_new_variant_cannot_load_old_weights(self):
        with patch("models.semantic_contour.FrozenSemanticPrior", FakePrior):
            old = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub")
            new = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub", contour_variant="selective_v1")
            same = SemanticContourGenerator(nn.Conv2d(3, 3, 1), "stub")
        legacy = dict(algorithm="structure_semantic_protection_v1", checkpoint_sha256="ab" * 32,
                      short_side=512, flip_tta=False, classes=IMPORTANT_CLASSES, settings=SETTINGS,
                      input="current transformed TIR; rounded uint8 for semantic inference",
                      cycle="recompute on current synthetic TIR; detached fixed condition")
        digest = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).digest()
        self.assertEqual(bytes(old.semantic_signature.tolist()), digest)
        same.load_state_dict(old.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
            new.load_state_dict(old.state_dict(), strict=True)


class CacheReplayTests(unittest.TestCase):
    def test_changed_inference_requires_full_numerical_replay_and_rejects_different_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "maps").mkdir()
            gray = np.full((8, 10), 127, np.uint8); Image.fromarray(gray).save(root / "source.png")
            p = scores("person", gray.shape)
            np.savez(root / "maps/01_responses.npz", TIR_uint8=gray, class_names=FMB_CLASS_NAMES,
                     base_probabilities=p, base_labels=p.argmax(0).astype(np.uint8))
            review = root / "review.json"; review.write_text("{}")
            protocol = root / "protocol.csv"; protocol.write_text("fixed")
            source_hash = sha256_file(root / "source.png")
            manifest = dict(settings=dict(short_side=512, baseline_flip_tta=False),
                frames=[dict(filename="source.png", prefix="01", sha256=source_hash)],
                review_manifest_sha256=sha256_file(review), protocol_manifest_sha256=sha256_file(protocol),
                code_hashes={"scripts/semantic_preview/inference.py": "outdated"})
            (root / "preview_manifest.json").write_text(json.dumps(manifest))
            (root / "status.json").write_text(json.dumps(dict(phase="completed", completed=1)))
            args = SimpleNamespace(semantic_preview_dir=root, input_dir=root,
                                   review_manifest=review, protocol_manifest=protocol)
            with patch("preview_semantic_contours.load_review", return_value=(
                    dict(source_hashes={"source.png": source_hash}), [dict(filename="source.png")], [])):
                with self.assertRaisesRegex(ValueError, "code changed"):
                    preflight(args)
                replay = unittest.mock.Mock(return_value=p)
                report = preflight(args, replay_probabilities=replay)[0]
                self.assertEqual(replay.call_count, 1)
                self.assertEqual(report["cache_revalidation"]["max_absolute_error"], 0)
                with self.assertRaises(AssertionError):
                    preflight(args, replay_probabilities=lambda _: scores("sky", gray.shape))


if __name__ == "__main__": unittest.main()
