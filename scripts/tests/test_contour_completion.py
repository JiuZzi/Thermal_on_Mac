"""Known-boundary completion checks, false-connection controls, review I/O."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pytorch-CycleGAN-and-pix2pix"))

from models.clarity_semantic_edges import soften_ridges
from models.contour_completion import (
    CompletionSettings, completion_comparison, evidence_fields, _render_paths, _anchored_thinning,
)
import preview_contour_completion as preview
from semantic_preview.checkpoint import sha256_file


def straight_fixture():
    image = np.full((64, 96), .2, np.float32)
    image[32:] = .8
    ridges = np.zeros(image.shape, bool)
    ridges[31, 8:88] = True
    ridges[31, 45:50] = False
    observed = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
    return image, observed, ridges


SETTINGS = CompletionSettings(max_gap_fraction=.12)


class CompletionTests(unittest.TestCase):
    def test_closing_thinning_keeps_original_ridges_instead_of_displacing_them(self):
        mask = np.zeros((32, 40), bool); mask[8:24, 10:30] = True
        anchors = np.zeros_like(mask); anchors[8, 10:30] = True; anchors[23, 10:30] = True
        result = _anchored_thinning(mask, anchors)
        self.assertTrue(np.all(result[anchors]))
        self.assertFalse(np.any(result & ~mask))
        self.assertLess(result.sum(), mask.sum())

    def test_local_circular_boundary_gap_stays_close_to_known_visible_boundary(self):
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import find_boundaries
        from skimage.morphology import thin
        y, x = np.indices((128, 128))
        inside = (y-64)**2 + (x-64)**2 <= 30**2
        image = np.where(inside, .8, .2).astype(np.float32)
        truth = thin(find_boundaries(inside, mode="inner"))
        ridges = truth.copy(); ridges[(y >= 60) & (y <= 66) & (x > 90)] = False
        observed = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
        maps, record = completion_comparison(image, observed, trace_ridges=ridges,
                                             settings=replace(SETTINGS, max_gap_fraction=.08))
        self.assertEqual(record["reliability_accepted"], 1)
        self.assertTrue(maps["reliable_mask"].any())
        self.assertLessEqual(float(distance_transform_edt(~truth)[maps["reliable_mask"]].max()), 1.5)

    def test_known_visible_straight_gap_is_recovered_in_all_completion_arms(self):
        image, observed, ridges = straight_fixture()
        maps, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        self.assertEqual(record["geometry_selected"], 1)
        self.assertEqual(record["reliability_accepted"], 1)
        for mode in ("closing", "geometry", "reliable"):
            self.assertTrue(maps[mode + "_mask"][31, 45:50].all(), mode)
            self.assertGreater(float(maps[mode][31, 47]), .7, mode)

    def test_original_input_is_not_mutated_or_weakened_and_outputs_are_finite(self):
        image, observed, ridges = straight_fixture()
        originals = [a.copy() for a in (image, observed, ridges)]
        maps, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        for actual, expected in zip((image, observed, ridges), originals):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(maps["observed"], observed)
        for mode in ("closing", "geometry", "reliable"):
            self.assertTrue(np.all(maps[mode] >= observed))
        for values in maps.values():
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((0 <= values) & (values <= 1)).all())
        json.dumps(record, allow_nan=False)

    def test_reliability_reuses_exact_selected_paths_and_is_pixelwise_bounded(self):
        image, observed, ridges = straight_fixture()
        maps, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        self.assertTrue(np.all(maps["reliable"] <= maps["geometry"]))
        self.assertFalse(np.any(maps["reliable_mask"] & ~maps["geometry_mask"]))
        for c in record["candidates"]:
            if c["reliability_accepted"]:
                self.assertTrue(c["geometry_selected"])
                self.assertEqual(c["path_yx"], [[31, x] for x in range(44, 51)])

    def test_flat_image_does_not_invent_boundaries_even_with_fake_fragment_anchors(self):
        _, observed, ridges = straight_fixture()
        maps, record = completion_comparison(np.full(observed.shape, .5, np.float32), observed,
                                             trace_ridges=ridges, settings=SETTINGS)
        self.assertEqual(record["geometry_selected"], 0)
        self.assertFalse(maps["geometry_mask"].any())
        self.assertGreater(record["pair_rejections"].get("anchor_image_conflict", 0), 0)

    def test_empty_and_uniform_images_remain_empty_without_forced_per_image_quota(self):
        for value in (0., .5, 1.):
            image = np.full((32, 40), value, np.float32)
            maps, record = completion_comparison(image, np.zeros_like(image))
            self.assertEqual(record["endpoints"], [])
            self.assertEqual(record["candidates"], [])
            for mode in ("observed", "closing", "geometry", "reliable"):
                self.assertFalse(maps[mode].any())
            self.assertFalse(evidence_fields(image)["strength"].any())

    def test_close_parallel_fragment_ends_are_not_connected_sideways(self):
        image = np.full((64, 96), .2, np.float32); image[30:] = .5; image[36:] = .8
        ridges = np.zeros(image.shape, bool); ridges[29, 10:46] = True; ridges[35, 10:46] = True
        observed = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
        maps, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        self.assertEqual(record["geometry_selected"], 0)
        self.assertGreater(record["pair_rejections"].get("endpoint_direction", 0), 0)
        self.assertFalse(maps["geometry_mask"].any())

    def test_gradient_with_wrong_orientation_does_not_certify_a_connection(self):
        _, observed, ridges = straight_fixture()
        image = np.full(observed.shape, .2, np.float32); image[:, 48:] = .8
        maps, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        self.assertEqual(record["geometry_selected"], 0)
        self.assertFalse(maps["geometry_mask"].any())

    def test_path_cannot_cross_an_existing_unrelated_boundary(self):
        image, _, ridges = straight_fixture()
        ridges[16:49, 47] = True
        observed = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
        maps, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        self.assertEqual(record["geometry_selected"], 0)
        self.assertFalse(maps["geometry_mask"].any())

    def test_gap_outside_the_recorded_scale_budget_is_not_filled(self):
        image, observed, ridges = straight_fixture()
        maps, record = completion_comparison(image, observed, trace_ridges=ridges)
        self.assertLess(record["gap_limit_pixels"], 6)
        self.assertEqual(record["geometry_selected"], 0)
        self.assertFalse(maps["geometry_mask"].any())

    def test_search_state_budget_is_reported_without_partial_path_output(self):
        image, observed, ridges = straight_fixture()
        maps, record = completion_comparison(image, observed, trace_ridges=ridges,
                                             settings=replace(SETTINGS, max_search_states=1))
        self.assertGreater(record["pair_rejections"].get("search_budget", 0), 0)
        self.assertFalse(maps["geometry_mask"].any())

    def test_pair_budget_truncation_is_explicit(self):
        image, _, ridges = straight_fixture()
        image[48:] = .2
        ridges[47, 8:88] = True; ridges[47, 45:50] = False
        observed = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
        _, record = completion_comparison(image, observed, trace_ridges=ridges,
                                         settings=replace(SETTINGS, max_searches=1))
        self.assertEqual(record["eligible_pairs"], 2)
        self.assertEqual(record["truncated_pairs"], 1)
        self.assertLessEqual(record["geometry_selected"], 1)

    def test_polarity_reversal_preserves_known_gap_geometry(self):
        image, observed, ridges = straight_fixture()
        first, _ = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        second, _ = completion_comparison(1-image, observed, trace_ridges=ridges, settings=SETTINGS)
        np.testing.assert_array_equal(first["geometry_mask"], second["geometry_mask"])
        np.testing.assert_allclose(first["reliable"], second["reliable"], atol=1e-6, rtol=0)

    def test_horizontal_flip_restores_the_known_connection_in_original_coordinates(self):
        image, observed, ridges = straight_fixture()
        first, _ = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        second, _ = completion_comparison(image[:, ::-1], observed[:, ::-1],
                                          trace_ridges=ridges[:, ::-1], settings=SETTINGS)
        np.testing.assert_array_equal(first["geometry_mask"], second["geometry_mask"][:, ::-1])

    def test_score_rejection_does_not_change_geometry_candidates(self):
        image, observed, ridges = straight_fixture()
        image[:, 44:51] = .2
        ridges[31, 44:51] = False
        observed = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
        maps, record = completion_comparison(image, observed, trace_ridges=ridges,
                                             settings=replace(SETTINGS, max_gap_fraction=.14))
        self.assertEqual(record["geometry_selected"], 1)
        self.assertEqual(record["reliability_accepted"], 0)
        self.assertTrue(maps["geometry_mask"].any())
        self.assertFalse(maps["reliable_mask"].any())
        self.assertTrue(record["candidates"][0]["reliability_rejections"])

    def test_reliability_render_is_monotone_with_neighboring_different_amplitudes(self):
        candidates = [dict(geometry_selected=True, reliability_accepted=True, strength=.8,
                           reliability_score=.8, path_yx=[[10, x] for x in range(8, 17)]),
                      dict(geometry_selected=True, reliability_accepted=False, strength=.1,
                           reliability_score=.2, path_yx=[[12, x] for x in range(8, 17)])]
        geometry, reliable = _render_paths((24, 32), candidates, .5)
        self.assertTrue(np.all(reliable <= geometry))
        self.assertAlmostEqual(float(reliable[10, 12]), .64, places=6)

    def test_each_selected_endpoint_is_used_only_once(self):
        image, observed, ridges = straight_fixture()
        _, record = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        used = [e for c in record["candidates"] if c["geometry_selected"] for e in c["endpoints"]]
        self.assertEqual(len(used), len(set(used)))

    def test_completion_masks_record_new_centerlines_not_existing_gaussian_tails(self):
        image, observed, ridges = straight_fixture()
        maps, _ = completion_comparison(image, observed, trace_ridges=ridges, settings=SETTINGS)
        self.assertGreater(float(observed[31, 45]), 0)
        self.assertTrue(maps["geometry_mask"][31, 45])
        self.assertFalse(np.any(maps["geometry_mask"] & maps["trace_ridges"]))

    def test_invalid_numeric_settings_and_inputs_are_rejected(self):
        for setting in (dict(max_gap_fraction=float("nan")), dict(soft_width=0),
                        dict(max_searches=1.5), dict(max_searches=True),
                        dict(reliability_min_score=1), dict(max_endpoint_angle=90)):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                CompletionSettings(**setting)
        image, observed, ridges = straight_fixture()
        bad = image.copy(); bad[0, 0] = np.nan
        for source, contour, centerlines in ((bad, observed, ridges),
                (image, observed[:3], ridges), (image, observed, ridges.astype(np.float32))):
            with self.assertRaises(ValueError):
                completion_comparison(source, contour, trace_ridges=centerlines)
        with self.assertRaises(TypeError):
            completion_comparison(image, observed, settings={})
        with self.assertRaises(ValueError):
            completion_comparison(np.zeros((4, 4)), np.zeros((4, 4)))


class ReviewTests(unittest.TestCase):
    def test_empty_candidate_tables_still_have_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"empty.csv"
            preview.write_table(p, [], ("candidate_id", "boundary_label", "notes"))
            self.assertEqual(p.read_text(encoding="utf-8-sig").strip(), "candidate_id,boundary_label,notes")

    def test_cli_preserves_existing_output_and_forbids_test_set_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                preview.parse_args(["--output-dir", tmp])
        with self.assertRaises(SystemExit):
            preview.parse_args(["--input-dir", str(ROOT / "FLIR_datasets/testB")])

    def test_cached_ridges_must_be_binary_and_image_aligned(self):
        image, observed, ridges = straight_fixture()
        gray = np.rint(image*255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"map.npz"
            arrays = dict(TIR_uint8=gray, clarity_v1=observed,
                clarity_v1_fine_ridges=ridges.astype(np.float32),
                clarity_v1_coarse_ridges=ridges.astype(np.float32),
                raw_fine_ridges=ridges.astype(np.float32))
            np.savez(p, **arrays)
            expected, actual = preview._cached_maps(p, gray)
            np.testing.assert_array_equal(expected, observed)
            np.testing.assert_array_equal(actual, ridges)
            with self.assertRaises(ValueError): preview._cached_maps(p, np.zeros_like(gray))
            arrays["raw_fine_ridges"] = np.full(image.shape, .5, np.float32)
            np.savez(p, **arrays)
            with self.assertRaises(ValueError): preview._cached_maps(p, gray)

    def test_actual_review_writer_exports_lossless_float_maps_and_metadata(self):
        image, observed, ridges = straight_fixture()
        gray = np.rint(image*255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base/"clarity"; (source/"maps").mkdir(parents=True)
            inputs = base/"inputs"; inputs.mkdir(); Image.fromarray(gray).save(inputs/"source.png")
            cache = source/"maps/01_contours.npz"
            np.savez(cache, TIR_uint8=gray, clarity_v1=observed,
                clarity_v1_fine_ridges=ridges.astype(np.float32),
                clarity_v1_coarse_ridges=ridges.astype(np.float32), raw_fine_ridges=ridges.astype(np.float32))
            (source/"preview_manifest.json").write_text("{}")
            semantic = base/"semantic"; semantic.mkdir(); (semantic/"preview_manifest.json").write_text("{}")
            review = base/"review.json"; review.write_text("{}")
            protocol = base/"protocol.csv"; protocol.write_text("fixture")
            output = base/"out"
            args = type("Args", (), dict(output_dir=output, clarity_dir=source, semantic_preview_dir=semantic,
                input_dir=inputs, review_manifest=review, protocol_manifest=protocol, zoom=2))()
            selected = [dict(filename="source.png", cache_prefix="01", cache_sha256=sha256_file(cache),
                             tir_sha256=sha256_file(inputs/"source.png"), video_id="fixture", group="synthetic")]
            with patch.object(preview, "preflight", return_value=(selected, [])), \
                 patch.object(preview, "CompletionSettings", return_value=SETTINGS):
                preview.run(args)
            status = json.loads((output/"status.json").read_text())
            self.assertEqual(status["phase"], "completed")
            with np.load(output/"maps/01_completion.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["observed"], observed)
                self.assertTrue(np.all(arrays["reliable"] <= arrays["geometry"]))
            self.assertTrue((output/"frames/01_comparison.png").is_file())
            self.assertTrue((output/"diagnostics/01_candidates.svg").is_file())
            metadata = json.loads((output/"diagnostics/01_candidates.json").read_text())
            self.assertEqual(metadata["geometry_selected"], 1)
            self.assertTrue((output/"candidate_labels_template.csv").is_file())
            with patch.object(preview, "preflight") as check, self.assertRaises(FileExistsError):
                preview.run(args)
            check.assert_not_called()

    def test_failed_status_write_does_not_mask_original_processing_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), dict(output_dir=Path(tmp)/"out", clarity_dir=Path(tmp),
                semantic_preview_dir=Path(tmp), input_dir=Path(tmp), review_manifest=Path(tmp)/"review.json",
                protocol_manifest=Path(tmp)/"protocol.csv"))()
            selected = [dict(filename="missing.png", tir_sha256="not-present", cache_prefix="01")]
            def fake_sha(path):
                if Path(path).name == "missing.png":
                    raise ValueError("original processing failure")
                return "digest"
            with patch.object(preview, "preflight", return_value=(selected, [])), \
                 patch.object(preview, "sha256_file", side_effect=fake_sha), \
                 patch.object(preview, "write_json", side_effect=[None, OSError("full disk")]):
                with self.assertRaisesRegex(ValueError, "original processing failure"):
                    preview.run(args)


if __name__ == "__main__":
    unittest.main()
