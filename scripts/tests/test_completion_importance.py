"""Task weighting invariants and class/geometry/quality stress controls."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'pytorch-CycleGAN-and-pix2pix'), str(ROOT/'scripts/tests')]
from test_contour_completion import straight_fixture, SETTINGS
from models.contour_completion import completion_comparison
from models.completion_importance import ImportanceSettings, prioritize_completion
from models.semantic_edges import FMB_CLASS_NAMES
import preview_completion_importance as review


def one_class(name, shape):
    p = np.zeros((len(FMB_CLASS_NAMES), *shape), np.float32)
    p[FMB_CLASS_NAMES.index(name)] = 1
    return p


class ImportanceTests(unittest.TestCase):
    def setUp(self):
        image, self.observed, ridges = straight_fixture()
        self.old, self.metadata = completion_comparison(image, self.observed, trace_ridges=ridges, settings=SETTINGS)
        self.structure = np.full(image.shape, .6, np.float32)
        self.clutter = np.full(image.shape, .2, np.float32)
        self.q = np.ones(image.shape, np.float32)

    def apply(self, name=None, **kwargs):
        p = one_class(name, self.observed.shape) if name is not None else None
        return prioritize_completion(self.observed, self.metadata, self.structure, self.clutter,
                                     p, self.q if p is not None else None, **kwargs)

    def test_original_paths_scores_flags_and_inputs_remain_unchanged(self):
        source = json.dumps(self.metadata, sort_keys=True)
        arrays = [a.copy() for a in (self.observed, self.structure, self.clutter)]
        maps, record = self.apply("car")
        self.assertEqual(source, json.dumps(self.metadata, sort_keys=True))
        for a, b in zip((self.observed, self.structure, self.clutter), arrays): np.testing.assert_array_equal(a, b)
        for old, new in zip(self.metadata['candidates'], record['candidates']):
            for key, value in old.items(): self.assertEqual(value, new[key], key)
        np.testing.assert_array_equal(maps['baseline'], self.old['reliable'])

    def test_weights_only_attenuate_additions_and_preserve_original_E(self):
        maps, _ = self.apply("road")
        np.testing.assert_array_equal(maps['observed'], self.observed)
        self.assertTrue(np.all(maps['priority'] >= self.observed))
        self.assertTrue(np.all(maps['priority'] <= maps['baseline']))
        for value in maps.values(): self.assertTrue(np.isfinite(value).all())

    def test_mean_control_matches_actual_added_strength_after_max_fusion(self):
        maps, _ = self.apply("road")
        self.assertAlmostEqual(float(maps['priority_increment'].mean(dtype=np.float64)),
                               float(maps['uniform_increment'].mean(dtype=np.float64)), places=9)
        self.assertTrue(np.all(maps['uniform'] >= self.observed))

    def test_all_configured_task_classes_receive_priority_without_class_specific_shapes(self):
        _, road = self.apply("road")
        for name in ImportanceSettings().task_classes:
            with self.subTest(name=name):
                _, record = self.apply(name)
                self.assertAlmostEqual(record['candidates'][0]['importance_gain'], 1.)
                self.assertGreater(record['candidates'][0]['importance_gain'], road['candidates'][0]['importance_gain'])

    def test_zero_quality_is_exact_structural_fallback_regardless_of_wrong_semantics(self):
        base, _ = self.apply()
        for name in ("car", "road", "sky", "vegetation"):
            p = one_class(name, self.observed.shape)
            maps, _ = prioritize_completion(self.observed, self.metadata, self.structure, self.clutter,
                                             p, np.zeros_like(self.q))
            np.testing.assert_array_equal(base['priority'], maps['priority'])

    def test_confident_wrong_background_retains_a_structural_backup(self):
        _, record = self.apply("sky")
        c = record['candidates'][0]
        expected = .25 + .75 * .5 * .6 * .8
        self.assertAlmostEqual(c['importance_gain'], expected, places=6)
        self.assertGreater(c['importance_gain'], ImportanceSettings().minimum_gain)

    def test_non_task_region_boundary_is_retained_even_when_center_Q_is_eroded(self):
        p = one_class("sky", self.observed.shape)
        p[:, 31:] = 0; p[FMB_CLASS_NAMES.index("vegetation"), 31:] = 1
        q = self.q.copy(); q[30:33] = 0
        _, record = prioritize_completion(self.observed, self.metadata, self.structure, self.clutter, p, q)
        _, interior = self.apply("vegetation")
        self.assertGreater(record['candidates'][0]['importance_features']['semantic_boundary_support'], .9)
        self.assertGreater(record['candidates'][0]['importance_gain'], interior['candidates'][0]['importance_gain'])

    def test_no_target_scene_preserves_nonzero_addition_without_forced_new_targets(self):
        maps, record = self.apply("road")
        self.assertGreater(float(maps['priority_increment'].sum()), 0)
        self.assertEqual(len(record['candidates']), len(self.metadata['candidates']))

    def test_clutter_changes_relevance_without_changing_structural_C(self):
        _, clean = prioritize_completion(self.observed, self.metadata, self.structure, np.zeros_like(self.clutter))
        _, busy = prioritize_completion(self.observed, self.metadata, self.structure, np.ones_like(self.clutter))
        self.assertGreater(clean['candidates'][0]['importance_gain'], busy['candidates'][0]['importance_gain'])
        self.assertEqual(clean['candidates'][0]['reliability_score'], busy['candidates'][0]['reliability_score'])

    def test_semantic_channel_permutation_does_not_change_results(self):
        expected, _ = self.apply("car")
        names = tuple(reversed(FMB_CLASS_NAMES))
        p = one_class("car", self.observed.shape)[::-1].copy()
        actual, _ = prioritize_completion(self.observed, self.metadata, self.structure, self.clutter,
                                           p, self.q, class_names=names)
        np.testing.assert_array_equal(expected['priority'], actual['priority'])

    def test_explicit_custom_taxonomy_can_use_class_agnostic_boundaries(self):
        p = np.zeros((2, *self.observed.shape), np.float32)
        p[0, :31] = 1; p[1, 31:] = 1
        settings = ImportanceSettings(task_classes=())
        _, record = prioritize_completion(self.observed, self.metadata, self.structure, self.clutter,
            p, self.q, class_names=('new_material_A', 'new_material_B'), settings=settings)
        self.assertAlmostEqual(record['candidates'][0]['importance_gain'], 1.)

    def test_empty_candidates_and_zero_actual_increments_are_well_defined(self):
        meta = copy.deepcopy(self.metadata); meta['candidates'] = []
        maps, record = prioritize_completion(self.observed, meta, self.structure, self.clutter)
        np.testing.assert_array_equal(maps['priority'], self.observed)
        self.assertEqual(record['importance']['uniform_mean_control_gain'], 1.)
        maps, _ = prioritize_completion(np.ones_like(self.observed), self.metadata, self.structure, self.clutter)
        self.assertFalse(maps['priority_increment'].any())

    def test_rotation_does_not_introduce_position_or_horizon_preferences(self):
        expected, record = self.apply("car")
        rotated = copy.deepcopy(self.metadata)
        for c in rotated['candidates']:
            c['path_yx'] = [[self.observed.shape[1]-1-x, y] for y, x in c['path_yx']]
        actual, other = prioritize_completion(np.rot90(self.observed), rotated, np.rot90(self.structure), np.rot90(self.clutter),
            np.rot90(one_class('car',self.observed.shape),axes=(1,2)), np.rot90(self.q))
        self.assertEqual(record['candidates'][0]['importance_gain'], other['candidates'][0]['importance_gain'])
        np.testing.assert_allclose(np.rot90(expected['priority']), actual['priority'], atol=1e-7)

    def test_same_rule_is_used_at_doubled_resolution(self):
        _, original = self.apply("car")
        meta = copy.deepcopy(self.metadata)
        for c in meta['candidates']:
            y, first = c['path_yx'][0]; _, last = c['path_yx'][-1]
            c['path_yx'] = [[2*y, x] for x in range(2*first, 2*last+1)]
        shape = (128, 192)
        _, scaled = prioritize_completion(np.zeros(shape,np.float32), meta,
            np.full(shape,.6,np.float32), np.full(shape,.2,np.float32), one_class('car',shape), np.ones(shape,np.float32))
        self.assertEqual(original['candidates'][0]['importance_gain'], scaled['candidates'][0]['importance_gain'])

    def test_bad_inputs_and_corrupt_candidate_records_are_rejected(self):
        for changes in ({'minimum_gain':0},{'structural_backup':float('nan')},{'task_classes':['car']},
                        {'task_classes':('car','car')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): ImportanceSettings(**changes)
        p = one_class('car',self.observed.shape); p[0,0,0] = np.nan
        with self.assertRaises(ValueError): prioritize_completion(self.observed,self.metadata,self.structure,self.clutter,p,self.q)
        with self.assertRaises(ValueError): prioritize_completion(self.observed,self.metadata,self.structure,self.clutter,quality=self.q)
        with self.assertRaises(ValueError): prioritize_completion(self.observed,self.metadata,self.structure,self.clutter,one_class('car',self.observed.shape))
        for change in ({'strength':float('nan')},{'path_yx':[[0,0],[1,1],[999,999]]},
                       {'geometry_selected':False,'reliability_accepted':True}):
            meta = copy.deepcopy(self.metadata); meta['candidates'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                prioritize_completion(self.observed,meta,self.structure,self.clutter)


class ReviewTests(unittest.TestCase):
    def test_external_selection_reads_only_train_folders_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ['MSRS/train/ir','MSRS/test/ir','LLVIP/infrared/train','LLVIP/infrared/test']:
                p=root/folder;p.mkdir(parents=True)
                for i in range(5):(p/f'{i}.png').write_bytes(f'{folder}/{i}'.encode())
            first,availability=review.select_external_training(root,2)
            second,_=review.select_external_training(root,2)
            self.assertEqual(first,second)
            self.assertEqual(len(first),4)
            self.assertTrue(all('/train/' in row['path'] and '/test/' not in row['path'] for row in first))
            self.assertEqual(availability['MSRS']['available_images'],5)
            self.assertEqual(review.select_external_training(root,0)[0],[])

    def test_cli_rejects_invalid_sample_budget_and_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):review.parse_args(['--output-dir',tmp])
            with self.assertRaises(SystemExit):review.parse_args(['--output-dir',str(Path(tmp)/'new'),'--external-samples','25'])

    def test_real_writer_saves_plan_before_outputs_and_preserves_same_paths(self):
        image,observed,ridges=straight_fixture()
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        shape=observed.shape
        data=(np.rint(image*255).astype(np.uint8),observed,old['reliable'],metadata,
              np.full(shape,.6,np.float32),np.full(shape,.2,np.float32),None,None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'preview_manifest.json').write_text('{}')
            args=type('Args',(),dict(output_dir=root/'out',completion_dir=root,
                external_samples=0,zoom=2))()
            row=dict(prefix='01',filename='fixture.png',dataset='FLIR-development',video_id='synthetic',semantic_mode='unit_no_semantics')
            with patch.object(review,'preflight',return_value=({'targets':[]},[row])), \
                 patch.object(review,'select_external_training',return_value=([],{})), \
                 patch.object(review,'load_development',return_value=data):
                review.run(args)
            out=args.output_dir
            self.assertTrue(json.loads((out/'frozen_plan.json').read_text())['parameters_fixed_before_processing'])
            status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['phase'],'completed')
            with np.load(out/'maps/01_importance.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['observed'],observed)
                np.testing.assert_array_equal(z['baseline'],old['reliable'])
                self.assertTrue(np.all(z['priority']<=z['baseline']))
            self.assertTrue((out/'semantic_proxy_statistics.csv').is_file())
            with self.assertRaises(FileExistsError):review.run(args)


if __name__ == '__main__':
    unittest.main()
