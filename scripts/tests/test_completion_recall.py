"""Observed weak-boundary recall, preservation and false-extension controls."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))

from models.clarity_semantic_edges import soften_ridges
from models.contour_completion import CompletionSettings, completion_comparison
from models.completion_recall import RecallSettings, recall_completion, source_diagnostics, _score
from models.completion_importance import prioritize_completion
from completion_recall_audit import (select_erasure, erasure_mask, control_statistics,
    annotation_statistics, read_annotations, candidate_mask)
import preview_completion_recall as preview
from semantic_preview.checkpoint import sha256_file


SETTINGS=CompletionSettings(max_gap_fraction=.12)


def fixture(contrast=.6):
    image=np.full((96,128),.2,np.float32);image[48:]+=contrast
    ridges=np.zeros(image.shape,bool);ridges[47,20:80]=True
    observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
    maps,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
    return image,observed,maps['trace_ridges'],maps,metadata


class RecallTests(unittest.TestCase):
    def test_recovers_measured_continuation_with_no_second_original_endpoint(self):
        image,observed,ridges,old,metadata=fixture(.04)
        new,updated=recall_completion(image,observed,ridges,metadata)
        self.assertEqual(metadata['reliability_accepted'],0)
        self.assertGreater(updated['weak_recall']['new_reliability_accepted'],0)
        self.assertTrue(new['recall_mask'][46:50,80:90].any())
        self.assertGreater(float((new['recalled']-new['baseline']).sum()),0)
        self.assertTrue(np.all(np.abs(np.argwhere(new['recall_mask'])[:,0]-47)<=1))

    def test_all_original_inputs_paths_and_C_are_preserved(self):
        image,observed,ridges,old,metadata=fixture()
        # Include a real legacy paired connection as well as open ends.
        ridges[47,45:50]=False
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        originals=[a.copy() for a in (image,observed,old['trace_ridges'])]
        frozen=json.dumps(metadata,sort_keys=True)
        new,updated=recall_completion(image,observed,old['trace_ridges'],metadata)
        self.assertGreater(metadata['reliability_accepted'],0)
        self.assertEqual(frozen,json.dumps(metadata,sort_keys=True))
        self.assertEqual(updated['candidates'][:len(metadata['candidates'])],metadata['candidates'])
        self.assertEqual(updated['endpoints'][:len(metadata['endpoints'])],metadata['endpoints'])
        for a,b in zip((image,observed,old['trace_ridges']),originals):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(old['reliable'],new['baseline'])
        self.assertTrue(np.all(new['recalled']>=new['baseline']))

    def test_original_R_and_observed_E_remain_unchanged_after_appending_proposals(self):
        image,observed,ridges,old,metadata=fixture()
        structure=np.full_like(image,.5);clutter=np.zeros_like(image)
        baseline,base_meta=prioritize_completion(observed,metadata,structure,clutter)
        _,updated=recall_completion(image,observed,ridges,metadata)
        recalled,new_meta=prioritize_completion(observed,updated,structure,clutter)
        self.assertEqual(new_meta['candidates'][:len(metadata['candidates'])],base_meta['candidates'])
        self.assertTrue(np.all(recalled['priority']>=baseline['priority']))
        self.assertTrue(np.all(recalled['priority']>=observed))

    def test_exact_same_C_formula_reproduces_the_old_known_gap_score(self):
        image,observed,ridges,old,metadata=fixture()
        ridges[47,45:50]=False
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        _,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        candidate=next(c for c in metadata['candidates'] if c['reliability_accepted'])
        checked=copy.deepcopy(candidate);checked['reliability_rejections']=[]
        _score(checked,[],SETTINGS)
        self.assertEqual(checked['reliability_score'],candidate['reliability_score'])
        self.assertEqual(checked['reliability_rejections'],candidate['reliability_rejections'])

    def test_flat_input_does_not_extend_even_with_strong_fake_observed_anchors(self):
        _,observed,ridges,_,_=fixture()
        image=np.full_like(observed,.5)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        new,updated=recall_completion(image,observed,old['trace_ridges'],metadata)
        self.assertEqual(updated['weak_recall']['new_candidates'],0)
        np.testing.assert_array_equal(new['recalled'],old['reliable'])

    def test_below_noise_floor_contrast_is_not_stretched_into_a_new_edge(self):
        image,observed,ridges,_,metadata=fixture(.0001)
        new,updated=recall_completion(image,observed,ridges,metadata)
        self.assertEqual(updated['weak_recall']['new_candidates'],0)
        self.assertFalse(new['recall_mask'].any())

    def test_without_original_anchors_a_full_input_edge_does_not_spawn_new_objects(self):
        image,_,_,_,_=fixture()
        observed=np.zeros_like(image);ridges=np.zeros_like(image,bool)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        new,updated=recall_completion(image,observed,ridges,metadata)
        self.assertEqual(updated['weak_recall']['new_candidates'],0)
        np.testing.assert_array_equal(new['recalled'],observed)

    def test_transverse_edge_cannot_authorize_an_outward_extension(self):
        _,observed,ridges,_,_=fixture()
        image=np.full_like(observed,.2);image[:,80:]=.8
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        new,updated=recall_completion(image,observed,old['trace_ridges'],metadata)
        self.assertEqual(updated['weak_recall']['new_reliability_accepted'],0)
        self.assertFalse(new['recall_mask'].any())

    def test_existing_third_boundary_and_image_border_are_not_crossed(self):
        image,observed,ridges,_,_=fixture()
        ridges[10:85,85]=True
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        new,updated=recall_completion(image,observed,old['trace_ridges'],metadata)
        self.assertFalse(new['recall_mask'][47,86:100].any())
        self.assertFalse(np.any(new['recall_mask']&old['trace_ridges']))
        self.assertFalse(new['recall_mask'][[0,-1],:].any())
        self.assertFalse(new['recall_mask'][:,[0,-1]].any())

    def test_new_candidates_stay_local_bounded_and_have_valid_provenance(self):
        image,observed,ridges,_,metadata=fixture()
        new,updated=recall_completion(image,observed,ridges,metadata)
        for c in updated['candidates'][len(metadata['candidates']):]:
            self.assertEqual(c['source'],'anchored_weak_observation')
            self.assertLessEqual(c['length'],metadata['gap_limit_pixels']+1e-8)
            self.assertLessEqual(c['strength'],metadata['endpoints'][c['endpoints'][0]]['strength'])
            points=np.asarray(c['path_yx'])[1:-1]
            self.assertTrue(new['weak_ridges'][tuple(points.T)].all())
        self.assertTrue(np.all(new['recalled']<=1))

    def test_each_original_endpoint_is_used_at_most_once_in_combined_geometry(self):
        image,observed,ridges,_,metadata=fixture()
        _,updated=recall_completion(image,observed,ridges,metadata)
        used=[i for c in updated['candidates'] if c['geometry_selected'] for i in c['endpoints']]
        self.assertEqual(len(used),len(set(used)))

    def test_original_rejected_geometry_remains_locked_not_silently_rematched(self):
        image,observed,ridges,_,_=fixture()
        ridges[47,45:50]=False;image[:,44:51]=.2
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        self.assertGreater(metadata['geometry_selected'],metadata['reliability_accepted'])
        _,updated=recall_completion(image,observed,old['trace_ridges'],metadata)
        locked={i for c in metadata['candidates'] if c['geometry_selected'] for i in c['endpoints']}
        for e in updated['weak_recall']['endpoint_ledger']:
            if e['endpoint_id'] in locked:self.assertEqual(e['status'],'legacy_geometry_locked')

    def test_polarity_reversal_preserves_recall_geometry(self):
        image,observed,ridges,old,metadata=fixture()
        first,_=recall_completion(image,observed,ridges,metadata)
        _,inverted_metadata=completion_comparison(1-image,observed,trace_ridges=ridges,settings=SETTINGS)
        second,_=recall_completion(1-image,observed,ridges,inverted_metadata)
        np.testing.assert_array_equal(first['recall_mask'],second['recall_mask'])
        np.testing.assert_allclose(first['recalled'],second['recalled'],atol=2e-6)

    def test_rotation_has_no_horizon_or_class_specific_rule(self):
        image,observed,ridges,_,metadata=fixture()
        first,_=recall_completion(image,observed,ridges,metadata)
        rotated=[np.rot90(v).copy() for v in (image,observed,ridges)]
        old,rotated_metadata=completion_comparison(rotated[0],rotated[1],trace_ridges=rotated[2],settings=SETTINGS)
        second,_=recall_completion(rotated[0],rotated[1],old['trace_ridges'],rotated_metadata)
        # NMS grid tie handling can differ by one pixel after a 90-degree turn.
        from scipy.ndimage import distance_transform_edt
        self.assertTrue(second['recall_mask'].any())
        self.assertLessEqual(float(distance_transform_edt(~np.rot90(first['recall_mask']))[second['recall_mask']].max()),1.5)

    def test_resolution_change_uses_relative_local_length_not_fixed_object_size(self):
        image,observed,ridges,_,metadata=fixture()
        first,record1=recall_completion(image,observed,ridges,metadata)
        image2=np.repeat(np.repeat(image,2,0),2,1)
        r2=np.zeros_like(image2,bool);r2[95,40:160]=True
        o2=soften_ridges(r2,np.full(image2.shape,.8,np.float32),.5)
        old,meta2=completion_comparison(image2,o2,trace_ridges=r2,settings=SETTINGS)
        second,record2=recall_completion(image2,o2,old['trace_ridges'],meta2)
        self.assertTrue(second['recall_mask'].any())
        self.assertGreater(int(second['recall_mask'].sum()),int(first['recall_mask'].sum()))
        self.assertEqual(meta2['gap_limit_pixels'],2*metadata['gap_limit_pixels'])

    def test_budgets_are_explicit_and_deterministic(self):
        image,observed,ridges,_,metadata=fixture()
        settings=RecallSettings(max_anchors=1,max_states_per_anchor=1)
        a,ma=recall_completion(image,observed,ridges,metadata,settings=settings)
        b,mb=recall_completion(image,observed,ridges,metadata,settings=settings)
        self.assertEqual(ma,mb)
        self.assertGreater(ma['weak_recall']['endpoint_status_counts'].get('anchor_budget',0),0)
        self.assertTrue(any(e.get('stops',{}).get('state_budget',0) for e in ma['weak_recall']['endpoint_ledger']))
        np.testing.assert_array_equal(a['recalled'],b['recalled'])

    def test_invalid_inputs_and_settings_fail_before_generating_paths(self):
        for options in (dict(beam_width=True),dict(max_anchors=1.5),dict(minimum_scale_agreement=float('nan')),
                        dict(minimum_length_fraction=0),dict(minimum_length_fraction=.2)):
            with self.subTest(options=options),self.assertRaises(ValueError):RecallSettings(**options)
        image,observed,ridges,_,metadata=fixture()
        for bad_ridges in (ridges.astype(np.float32),ridges[:3]):
            with self.assertRaises(ValueError):recall_completion(image,observed,bad_ridges,metadata)
        corrupt=copy.deepcopy(metadata);corrupt['endpoints'][0]['point']=[-1,0]
        with self.assertRaises(ValueError):recall_completion(image,observed,ridges,corrupt)
        with self.assertRaises(TypeError):recall_completion(image,observed,ridges,metadata,settings={})
        _,updated=recall_completion(image,observed,ridges,metadata)
        with self.assertRaises(ValueError):recall_completion(image,observed,ridges,updated)


class AuditTests(unittest.TestCase):
    def test_erasure_selection_is_deterministic_and_does_not_consume_model_output(self):
        _,observed,ridges,_,_=fixture()
        a=select_erasure(ridges,observed,42,10);b=select_erasure(ridges,observed,42,10)
        self.assertIsNotNone(a);self.assertEqual(a,b);self.assertEqual(len(a),10)
        self.assertTrue(ridges[tuple(np.asarray(a).T)].all())
        mask=erasure_mask(ridges.shape,a)
        self.assertTrue(mask[tuple(np.asarray(a).T)].all())
        self.assertIsNone(select_erasure(np.zeros_like(ridges),observed,42,10))

    def test_auxiliary_controls_distinguish_no_candidate_from_weak_output(self):
        image,observed,ridges,old,metadata=fixture()
        new,updated=recall_completion(image,observed,ridges,metadata)
        truth=new['recall_mask'];erased=truth.copy()
        stats=control_statistics(truth,erased,truth,metadata,updated,old['reliable']-observed,new['recalled']-observed)
        self.assertEqual(stats['baseline_all_coverage_1px'],0)
        self.assertEqual(stats['recall_reliable_coverage_1px'],1)
        self.assertEqual(stats['recall_off_reference_pixels_inside_erasure'],0)
        self.assertIn('recall_unlabelled_addition_pixels_outside_erasure',stats)

    def test_manual_annotations_keep_negative_and_uncertain_examples_separate(self):
        image,observed,ridges,old,metadata=fixture()
        new,updated=recall_completion(image,observed,ridges,metadata)
        for label in ('visible_boundary','no_boundary','uncertain'):
            gap=dict(id='g1',prefix='01',label=label,polyline_yx=[[47,80],[47,86]])
            row=annotation_statistics(gap,image.shape,metadata,updated,old['reliable']-observed,new['recalled']-observed)
            self.assertEqual(row['label'],label)
            self.assertEqual(row['baseline_bottleneck_at_half_coverage'],'no_candidate')
            self.assertGreater(row['recall_all_coverage_1px'],0)
        gap['polyline_yx']=[[-1,0],[2,3]]
        with self.assertRaises(ValueError):annotation_statistics(gap,image.shape,metadata,updated,observed,observed)

    def test_annotation_loader_rejects_unknown_frames_duplicate_ids_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.json'
            gap=dict(id='g1',prefix='01',label='visible_boundary',polyline_yx=[[1,2],[3,4]])
            def save(gaps):path.write_text(json.dumps(dict(format='completion_gap_annotations_v1',gaps=gaps)))
            save([gap]);self.assertEqual(read_annotations(path,{'01'}),[gap])
            save([gap,gap])
            with self.assertRaises(ValueError):read_annotations(path,{'01'})
            save([gap])
            with self.assertRaises(ValueError):read_annotations(path,{'02'})
            save([dict(gap,polyline_yx=[[float('nan'),2],[3,4]])])
            with self.assertRaises(ValueError):read_annotations(path,{'01'})
        self.assertEqual(read_annotations(None,{'01'}),[])

    def test_stage_counts_do_not_mislabel_path_rejections_as_accuracy(self):
        _,_,_,_,metadata=fixture()
        result=source_diagnostics(metadata)
        self.assertEqual(result['admissible_paths'],0)
        self.assertIn('no independent',result['interpretation'])

    def test_cli_preserves_outputs_and_validates_review_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):preview.parse_args(['--output-dir',tmp])
        for args in (['--erasure-samples','41'],['--external-samples','-1'],['--zoom','0']):
            with self.assertRaises(SystemExit):preview.parse_args(args)
        with self.assertRaises(SystemExit):preview.parse_args(['--input-dir',str(ROOT/'FLIR_datasets/testB')])

    def test_actual_writer_preserves_float_maps_and_declares_missing_real_gap_labels(self):
        image,observed,ridges,old,metadata=fixture()
        gray=np.rint(image*255).astype(np.uint8);image=preview._image01(gray)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        structure=np.full_like(image,.5);clutter=np.zeros_like(image)
        old_R,old_updated=prioritize_completion(observed,metadata,structure,clutter)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);importance=root/'previous';(importance/'maps').mkdir(parents=True)
            (importance/'diagnostics').mkdir()
            np.savez(importance/'maps/01_importance.npz',observed=observed,priority=old_R['priority'])
            (importance/'diagnostics/01_paths.json').write_text(json.dumps(old_updated))
            (importance/'run_manifest.json').write_text(json.dumps(dict(plan=dict(code_sha256={}))))
            row=dict(prefix='01',filename='source.png',dataset='FLIR-development',video_id='v1',semantic_mode='none')
            args=type('Args',(),dict(output_dir=root/'output',importance_dir=importance,external_samples=0,erasure_samples=0,annotations=None,zoom=2))()
            data=(gray,observed,old['reliable'],metadata,structure,clutter,None,None,old['trace_ridges'])
            identity=dict(maps_sha256={'01':sha256_file(importance/'maps/01_importance.npz')},
                diagnostics_sha256={'01':sha256_file(importance/'diagnostics/01_paths.json')})
            with patch.object(preview,'preflight',return_value=(dict(targets=[]),[row])),\
                 patch.object(preview,'select_external_training',return_value=([],{})),\
                 patch.object(preview,'source_identity',return_value=identity),\
                 patch.object(preview,'control_plan',return_value=([],{},[])),\
                 patch.object(preview,'verify_source_files'),\
                 patch.object(preview,'load_input',return_value=data):
                preview.run(args)
            self.assertTrue((args.output_dir/'frozen_plan.json').exists())
            self.assertTrue((args.output_dir/'annotate_gaps.html').exists())
            self.assertTrue((args.output_dir/'endpoint_diagnostics.csv').exists())
            with np.load(args.output_dir/'maps/01_recall.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['observed'],observed)
                np.testing.assert_array_equal(z['baseline'],old_R['priority'])
                self.assertTrue((z['recall']>=z['baseline']).all())
            status=json.loads((args.output_dir/'status.json').read_text())
            self.assertEqual(status['phase'],'completed')
            self.assertEqual(status['real_gap_annotations'],0)
            self.assertFalse(status['real_gap_quality_verified'])
            self.assertEqual((args.output_dir/'gap_statistics.csv').read_text(encoding='utf-8-sig').strip(),'id,prefix,label,notes')


if __name__=='__main__':unittest.main()
