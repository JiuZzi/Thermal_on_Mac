"""Analytic boundaries/negatives, immutable provenance, and staged selection."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pytorch-CycleGAN-and-pix2pix'))
sys.path.insert(0, str(ROOT / 'scripts'))
from models.contour_completion import CompletionSettings, completion_comparison, _path_features, evidence_fields
from models.clarity_semantic_edges import soften_ridges
from models.completion_importance import prioritize_completion
from models.region_completion import RegionSettings, ARMS, region_evidence, completion_stages, path_key


def fixture(*, observations=True, duplicate=False, weak=False):
    image = np.full((96, 128), .2, np.float32); image[48:] = .8
    ridges = np.zeros(image.shape, bool)
    ridges[47, 20:59] = True; ridges[47, 68:110] = True
    E = soften_ridges(ridges, np.full(image.shape, .8, np.float32), .5)
    settings = CompletionSettings(max_gap_fraction=.12)
    _, original = completion_comparison(image, E, trace_ridges=ridges, settings=settings)
    # Isolate a new proposal from old matching. This is not boundary ground truth.
    original['candidates'] = []
    original['geometry_selected'] = original['reliability_accepted'] = 0
    endpoints = original['endpoints']
    a = next(e for e in endpoints if e['point'] == [47, 58])
    b = next(e for e in endpoints if e['point'] == [47, 68])
    path = [[47, x] for x in range(58, 69)]
    f = _path_features(np.asarray(path), a, b, evidence_fields(image), settings)
    if not observations:
        # Explicit input to the decision-rule test: weak centerline measurement.
        f.update(mean_support=0., support_coverage=0., cross_scale_agreement=0., unsupported_run=1.)
    c = dict(id=0, endpoints=[a['id'], b['id']], path_yx=path, gap=10., length=10., cost=10.,
        strength=.8, features=f, geometry_selected=False, reliability_accepted=False,
        geometry_rejection=None, reliability_rejections=['old_test_rejection'],
        reliability_score=.1, source='two_measured_fragments')
    fragment = copy.deepcopy(original)
    fragment['candidates'] = [c]
    if duplicate:
        fragment['candidates'].append(dict(copy.deepcopy(c), id=1))
    fragment['fragment_completion'] = dict(format='two_fragment_completion_v2', original_candidates=0)
    recalled = copy.deepcopy(original)
    recalled['weak_recall'] = dict(format='anchored_weak_recall_v1', original_candidates=0)
    if weak:
        c = copy.deepcopy(c); c.update(source='anchored_weak_observation', closing_original_endpoint=True)
        recalled['candidates'] = [c]
    return image, E, ridges, original, fragment, recalled


class RegionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.path = [[47, x] for x in range(58, 69)]
        self.image = np.full((96, 128), .2, np.float32); self.image[48:] = .8

    def test_piecewise_constant_boundary_beats_no_boundary_on_held_out_pixels(self):
        r = region_evidence(self.image, [self.path])[0]
        self.assertTrue(r['available']); self.assertGreater(r['support'], .9)
        self.assertGreater(r['null_loss'], r['split_loss'] + r['complexity_penalty'])

    def test_flat_and_smooth_thermal_ramps_prefer_abstention(self):
        y, x = np.indices(self.image.shape)
        for image in (np.full_like(self.image, .5), (.2 + .001*y + .002*x).astype(np.float32)):
            r = region_evidence(image, [self.path])[0]
            self.assertEqual(r['support'], 0.)
            self.assertLess(r['signed_improvement'], 0.)

    def test_noise_is_not_an_extra_region_by_default(self):
        image = np.random.default_rng(613).normal(.5, .035, self.image.shape).clip(0, 1).astype(np.float32)
        paths = [[[y, x] for x in range(30, 41)] for y in range(20, 70, 5)]
        scores = [r['support'] for r in region_evidence(image, paths)]
        self.assertLess(max(scores), RegionSettings().minimum_region_support)

    def test_noise_does_not_hide_a_strong_boundary(self):
        image = (self.image + np.random.default_rng(91).normal(0, .02, self.image.shape)).clip(0, 1).astype(np.float32)
        self.assertGreater(region_evidence(image, [self.path])[0]['support'], .5)

    def test_strong_endpoints_do_not_certify_flat_gap_interior(self):
        image = self.image.copy(); image[:, 58:69] = .5
        self.assertEqual(region_evidence(image, [self.path])[0]['support'], 0.)

    def test_reversed_path_and_thermal_polarity_keep_same_evidence(self):
        a = region_evidence(self.image, [self.path])[0]
        for image, path in ((self.image, self.path[::-1]), (1-self.image, self.path)):
            b = region_evidence(image, [path])[0]
            for name in ('support', 'null_loss', 'split_loss'):
                self.assertAlmostEqual(a[name], b[name], places=6)

    def test_ninety_degree_rotation_uses_no_horizon_rule(self):
        path = [[self.image.shape[1]-1-x, y] for y, x in self.path]
        a = region_evidence(self.image, [self.path])[0]
        b = region_evidence(np.rot90(self.image).copy(), [path])[0]
        self.assertAlmostEqual(a['support'], b['support'], places=6)
        self.assertEqual(a['unique_pixels'], b['unique_pixels'])

    def test_batch_size_changes_computation_only(self):
        paths = [self.path, self.path[::-1], [[46, x] for x in range(58, 69)]]
        a = region_evidence(self.image, paths, settings=replace(RegionSettings(), batch_size=1))
        b = region_evidence(self.image, paths)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x['support'], y['support'], places=10)

    def test_short_or_border_bands_are_unknown_not_positive(self):
        paths = [[[0, x] for x in range(4, 8)], [[47, 58], [47, 59], [47, 60]]]
        for r in region_evidence(self.image, paths):
            self.assertFalse(r['available']); self.assertEqual(r['support'], 0.)

    def test_all_values_are_finite_JSON_and_duplicate_samples_do_not_inflate_n(self):
        r = region_evidence(self.image, [self.path])[0]
        self.assertLessEqual(r['unique_pixels'], (len(self.path)-2) * 2*r['band_radius'])
        self.assertGreater(r['complexity_penalty'], 0.)
        json.dumps(r, allow_nan=False)


class SelectionTests(unittest.TestCase):
    def test_exact_duplicates_are_merged_and_all_source_records_survive(self):
        values = fixture(duplicate=True)
        outputs, audit = completion_stages(*values)
        self.assertEqual(outputs['dedup']['region_completion']['exact_duplicate_records'], 1)
        self.assertEqual(len(outputs['dedup']['candidates']), 1)
        self.assertEqual(len(audit['groups'][0]['members']), 2)
        self.assertEqual(audit['groups'][0]['members'][0]['reliability_score'], .1)
        self.assertEqual(outputs['dedup']['candidates'][0]['features']['alternative_margin'], 1.)

    def test_reversed_duplicate_does_not_become_a_competitor(self):
        values = fixture(duplicate=True)
        second = values[4]['candidates'][1]
        second['path_yx'].reverse(); second['endpoints'].reverse()
        outputs, audit = completion_stages(*values)
        self.assertEqual(outputs['dedup']['region_completion']['exact_duplicate_records'], 1)
        self.assertNotEqual(audit['groups'][0]['members'][0]['path_reversed_from_canonical'], audit['groups'][0]['members'][1]['path_reversed_from_canonical'])

    def test_regional_route_can_use_non_centerline_evidence(self):
        outputs, _ = completion_stages(*fixture(observations=False))
        self.assertFalse(outputs['dedup']['candidates'][0]['reliability_accepted'])
        c = outputs['region']['candidates'][0]
        self.assertTrue(c['reliability_accepted']); self.assertEqual(c['evidence_route'], 'regional_gap')
        self.assertFalse(outputs['observed_control']['candidates'][0]['reliability_accepted'])

    def _competing_fixture(self):
        values = fixture()
        fragment = values[4]
        endpoint = dict(copy.deepcopy(fragment['endpoints'][fragment['candidates'][0]['endpoints'][1]]),
            id=len(fragment['endpoints']), point=[46, 68])
        fragment['endpoints'].append(endpoint)
        c = copy.deepcopy(fragment['candidates'][0]); c['id'] = 1
        c['endpoints'][1] = endpoint['id']
        c['path_yx'] = [[47, x] for x in range(58, 64)] + [[46, x] for x in range(64, 69)]
        p = np.asarray(c['path_yx']); c['gap'] = float(np.linalg.norm(p[-1]-p[0]))
        c['length'] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        fragment['candidates'].append(c)
        return values

    def test_two_equally_supported_distinct_connections_can_both_abstain(self):
        outputs, _ = completion_stages(*self._competing_fixture())
        cs = outputs['region']['candidates']
        self.assertEqual(len(cs), 2)
        self.assertTrue(all(not c['reliability_accepted'] for c in cs))
        self.assertTrue(all('ambiguous_connection' in c['reliability_rejections'] for c in cs))

    def test_rejected_low_cost_candidate_does_not_occupy_a_new_site(self):
        values = self._competing_fixture()
        bad = values[4]['candidates'][1]
        bad.update(cost=1., source='anchored_weak_observation', closing_original_endpoint=False)
        bad['features'].update(mean_support=0., support_coverage=0., cross_scale_agreement=0., unsupported_run=1.)
        outputs, _ = completion_stages(*values)
        good = next(c for c in outputs['region']['candidates'] if c['proposal_id'] == 0)
        bad = next(c for c in outputs['region']['candidates'] if c['proposal_id'] == 1)
        self.assertTrue(good['reliability_accepted'])
        self.assertFalse(bad['geometry_selected'])
        self.assertEqual(bad['geometry_rejection'], 'C_rejected_before_matching')

    def test_empty_gap_without_regional_evidence_abstains(self):
        values = list(fixture(observations=False)); values[0] = np.full_like(values[0], .5)
        outputs, _ = completion_stages(*values)
        self.assertFalse(outputs['region']['candidates'][0]['reliability_accepted'])
        self.assertEqual(outputs['region']['candidates'][0]['evidence_route'], 'abstain')

    def test_one_observed_anchor_never_becomes_a_two_sided_gap_prior(self):
        values = fixture(observations=False, weak=True)
        values[5]['candidates'][0]['closing_original_endpoint'] = False
        outputs, _ = completion_stages(*values)
        self.assertFalse(outputs['weak_region']['candidates'][0]['reliability_accepted'])
        self.assertFalse(outputs['weak_region']['candidates'][0]['anchor_evidence']['two_measured_anchors'])

    def test_weak_and_fragment_records_share_one_geometric_hypothesis(self):
        outputs, audit = completion_stages(*fixture(weak=True))
        self.assertEqual(outputs['joint']['region_completion']['exact_duplicate_records'], 1)
        self.assertEqual(len(outputs['joint']['candidates']), 1)
        self.assertEqual({m['pool'] for m in audit['groups'][0]['members']}, {'fragment', 'weak'})

    def test_original_C_and_input_arrays_are_immutable_and_R_remains_separate(self):
        values = fixture(weak=True)
        before = json.dumps(values[3:], sort_keys=True)
        arrays = [v.copy() for v in values[:3]]
        outputs, _ = completion_stages(*values)
        self.assertEqual(before, json.dumps(values[3:], sort_keys=True))
        for a, b in zip(values[:3], arrays): np.testing.assert_array_equal(a, b)
        E = values[1]; structure = np.full_like(E, .5); clutter = np.zeros_like(E)
        for record in outputs.values():
            maps, _ = prioritize_completion(E, record, structure, clutter)
            self.assertTrue((maps['priority'] >= E).all())
            self.assertTrue(np.isfinite(maps['priority']).all())
            self.assertTrue((maps['priority'] <= 1).all())

    def test_original_real_completion_survives_all_arms(self):
        from models.fragment_completion import fragment_completion
        from models.completion_recall import recall_completion
        image, E, ridges, _, _, _ = fixture()
        _, original = completion_comparison(image, E, trace_ridges=ridges, settings=CompletionSettings(max_gap_fraction=.12))
        fragments, _, _ = fragment_completion(image, E, ridges, original)
        _, weak = recall_completion(image, E, ridges, original)
        outputs, _ = completion_stages(image, E, ridges, original, fragments, weak)
        for record in outputs.values():
            self.assertEqual(record['candidates'][:len(original['candidates'])], original['candidates'])

    def test_no_artificial_grid_loop_or_original_ridge_crossing_is_admitted(self):
        values = fixture()
        values[4]['candidates'][0]['path_yx'].insert(3, [47, 59])
        with self.assertRaises(ValueError): completion_stages(*values)
        values = fixture(); values[2][47, 63] = True
        with self.assertRaises(ValueError): completion_stages(*values)

    def test_wrong_source_pool_or_nonfinite_features_fail(self):
        values = fixture(); values[4]['fragment_completion']['original_candidates'] = 9
        with self.assertRaises(ValueError): completion_stages(*values)
        values = fixture(); values[4]['candidates'][0]['features']['mean_support'] = float('nan')
        with self.assertRaises(ValueError): completion_stages(*values)

    def test_invalid_map_and_settings_are_rejected(self):
        for kwargs in (dict(batch_size=0), dict(batch_size=True), dict(minimum_samples=2), dict(complexity_weight=0), dict(minimum_region_support=float('nan'))):
            with self.assertRaises(ValueError): RegionSettings(**kwargs)
        values = list(fixture()); values[0] = values[0].copy(); values[0][0, 0] = float('nan')
        with self.assertRaises(ValueError): completion_stages(*values)

    def test_empty_pools_remain_empty(self):
        values = fixture(); values[4]['candidates'] = []
        outputs, audit = completion_stages(*values)
        self.assertFalse(audit['groups'])
        self.assertTrue(all(not record['candidates'] for record in outputs.values()))

    def test_path_key_is_orientation_independent(self):
        p = [[2, 3], [2, 4], [3, 5]]
        self.assertEqual(path_key(p), path_key(p[::-1]))


class PreviewIntegrationTests(unittest.TestCase):
    def test_fresh_R_matches_the_original_importance_function_in_every_arm(self):
        from preview_region_completion import apply_stages
        image, E, ridges, original, fragments, weak = fixture(weak=True)
        gray = np.rint(image*255).astype(np.uint8)
        structure = np.full_like(E, .4); clutter = np.zeros_like(E)
        data = (gray, E, E, original, structure, clutter, None, None, ridges)
        maps, records, _, _ = apply_stages(data, fragments, weak)
        for arm in ARMS:
            expected, _ = prioritize_completion(E, records[arm], structure, clutter)
            np.testing.assert_array_equal(maps[arm], expected['priority'])

    def test_cached_and_fresh_importance_have_identical_float_outputs(self):
        from preview_region_completion import apply_stages
        image, E, ridges, original, fragments, weak = fixture(weak=True)
        gray = np.rint(image*255).astype(np.uint8)
        structure = np.full_like(E, .4); clutter = np.zeros_like(E)
        data = (gray, E, E, original, structure, clutter, None, None, ridges)
        a, _, _, _ = apply_stages(data, fragments, weak)
        _, base = prioritize_completion(E, original, structure, clutter)
        _, fragments = prioritize_completion(E, fragments, structure, clutter)
        _, weak = prioritize_completion(E, weak, structure, clutter)
        b, _, _, _ = apply_stages(data, fragments, weak, cached_baseline=base)
        for name in a: np.testing.assert_array_equal(a[name], b[name])

    def test_preview_preserves_existing_outputs_and_rejects_invalid_subsets(self):
        from preview_region_completion import parse_args
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit): parse_args(['--output-dir', tmp])
            with self.assertRaises(SystemExit): parse_args(['--limit', '0'])
            with self.assertRaises(SystemExit): parse_args(['--limit', '53'])

    def test_staged_comparisons_use_distinct_portable_filenames(self):
        from preview_region_completion import save_sheets, GROUPS
        image, E, ridges, original, fragments, weak = fixture()
        gray = np.rint(image*255).astype(np.uint8)
        from preview_region_completion import apply_stages
        data = (gray, E, E, original, np.full_like(E, .4), np.zeros_like(E), None, None, ridges)
        maps, _, _, _ = apply_stages(data, fragments, weak)
        maps['fragment'] = maps['baseline']; maps['weak'] = maps['baseline']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'frames').mkdir()
            save_sheets(root, 'fixture', gray, maps)
            self.assertEqual({p.name for p in (root/'frames').iterdir()}, {f'fixture_{group}.png' for group in GROUPS})


if __name__ == '__main__':
    unittest.main()
