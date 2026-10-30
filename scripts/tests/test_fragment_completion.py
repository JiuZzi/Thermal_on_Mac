"""Fragment/junction hypotheses, hard negatives, immutable C/R and rematching."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
sys.path.insert(0,str(ROOT/'scripts'))
from models.clarity_semantic_edges import soften_ridges
from models.contour_completion import completion_comparison,CompletionSettings,_render_paths
from models.completion_importance import prioritize_completion
from models.fragment_completion import (
    FragmentSettings,fragment_completion,rematch_fixed_C,_graph,_segments,_region_cost,
    sanitize_cached_fragments,
)
import preview_fragment_completion as preview
from semantic_preview.checkpoint import sha256_file

SETTINGS=CompletionSettings(max_gap_fraction=.12)


def junction_fixture():
    image=np.full((96,128),.2,np.float32);image[:,64:]=.7;image[50:]+=.08
    ridges=np.zeros(image.shape,bool);ridges[15:44,63]=True;ridges[50:85,63]=True;ridges[50,40:90]=True
    observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
    old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
    return image,observed,old['trace_ridges'],metadata


def paired_fixture():
    image=np.full((96,128),.2,np.float32);image[48:]=.8
    ridges=np.zeros(image.shape,bool);ridges[47,10:115]=True;ridges[47,60:66]=False
    observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
    old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
    return image,observed,old['trace_ridges'],metadata


class FragmentTests(unittest.TestCase):
    def test_junction_branch_supplies_an_anchor_that_old_endpoint_search_missed(self):
        image,observed,ridges,metadata=junction_fixture()
        self.assertEqual(metadata['reliability_accepted'],0)
        updated,probe,_=fragment_completion(image,observed,ridges,metadata)
        new=updated['candidates'][len(metadata['candidates']):]
        accepted=[c for c in new if c['reliability_accepted']]
        self.assertTrue(accepted)
        self.assertTrue(any(c['junction_anchor'] for c in accepted))
        self.assertTrue(any([46,63] in c['path_yx'] for c in accepted))

    def test_public_JSON_list_endpoints_work_with_original_tuple_array_path_search(self):
        image,observed,ridges,metadata=junction_fixture()
        metadata=json.loads(json.dumps(metadata))
        updated,_,_=fragment_completion(image,observed,ridges,metadata)
        self.assertGreater(updated['fragment_completion']['new_candidates'],0)
        json.dumps(updated,allow_nan=False)

    def test_original_inputs_candidates_and_accepted_C_R_are_preserved(self):
        image,observed,ridges,metadata=paired_fixture()
        before=json.dumps(metadata,sort_keys=True);arrays=[v.copy() for v in (image,observed,ridges)]
        new,_,_=fragment_completion(image,observed,ridges,metadata)
        self.assertEqual(before,json.dumps(metadata,sort_keys=True))
        self.assertEqual(new['candidates'][:len(metadata['candidates'])],metadata['candidates'])
        self.assertEqual(new['endpoints'][:len(metadata['endpoints'])],metadata['endpoints'])
        for a,b in zip((image,observed,ridges),arrays):np.testing.assert_array_equal(a,b)
        structure=np.full_like(image,.5);clutter=np.zeros_like(image)
        base,base_meta=prioritize_completion(observed,metadata,structure,clutter)
        after,after_meta=prioritize_completion(observed,new,structure,clutter)
        self.assertEqual(after_meta['candidates'][:len(metadata['candidates'])],base_meta['candidates'])
        self.assertTrue((after['priority']>=base['priority']).all())

    def test_short_gap_can_be_proposed_despite_missing_measured_interior(self):
        image,observed,ridges,metadata=junction_fixture()
        image[44:50,:]=.45
        _,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        updated,_,_=fragment_completion(image,observed,ridges,metadata)
        new=updated['candidates'][len(metadata['candidates']):]
        cross=[c for c in new if [46,63] in c['path_yx']]
        self.assertTrue(cross,'A gap hypothesis must reach C rather than be stopped by NMS')
        self.assertTrue(any(c['features']['support_coverage']<1 for c in cross))

    def test_strong_fragment_anchors_do_not_certify_a_long_flat_interior(self):
        image=np.full((288,360),.2,np.float32);image[:,180:]=.7;image[134:]+=.08
        image[110:134,:]=.45
        ridges=np.zeros(image.shape,bool);ridges[20:110,179]=True;ridges[134:265,179]=True;ridges[134,135:225]=True
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        original=CompletionSettings(max_gap_fraction=.12)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=original)
        updated,_,_=fragment_completion(image,observed,old['trace_ridges'],metadata)
        crossing=[c for c in updated['candidates'][len(metadata['candidates']):] if [122,179] in c['path_yx']]
        self.assertTrue(crossing)
        self.assertTrue(all(c['reliability_rejections'] for c in crossing))

    def test_observation_amplitude_is_a_separate_probe_on_identical_paths_C_and_R(self):
        image,observed,ridges,metadata=junction_fixture()
        new,probe,_=fragment_completion(image,observed,ridges,metadata)
        for a,b in zip(new['candidates'],probe['candidates']):
            self.assertEqual({k:v for k,v in a.items() if k!='strength'},{k:v for k,v in b.items() if k!='strength'})
        structure=np.full_like(image,.5);clutter=np.zeros_like(image)
        _,a=prioritize_completion(observed,new,structure,clutter)
        _,b=prioritize_completion(observed,probe,structure,clutter)
        self.assertEqual([c['add_importance'] for c in a['candidates']],[c['add_importance'] for c in b['candidates']])

    def test_flat_image_cannot_create_fragments_from_fake_strong_E(self):
        image,observed,ridges,_=junction_fixture();image[:]=.5
        _,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        updated,_,_=fragment_completion(image,observed,ridges,metadata)
        self.assertEqual(updated['fragment_completion']['new_candidates'],0)

    def test_one_measured_fragment_does_not_infer_a_whole_missing_object(self):
        image=np.full((96,128),.2,np.float32);image[:,64:]=.7
        ridges=np.zeros(image.shape,bool);ridges[15:85,63]=True
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        updated,_,_=fragment_completion(image,observed,old['trace_ridges'],metadata)
        self.assertEqual(updated['fragment_completion']['new_reliability_accepted'],0)

    def test_corner_diagonals_do_not_create_artificial_graph_junctions(self):
        ridges=np.zeros((12,12),bool);ridges[3,2:6]=True;ridges[3:9,5]=True
        graph=_graph(ridges)
        self.assertFalse(any(len(v)>2 for v in graph.values()))
        self.assertEqual(len(_segments(ridges)[0]),1)

    def test_closed_measured_loop_has_no_fabricated_ports(self):
        ridges=np.zeros((20,20),bool);ridges[4,4:16]=True;ridges[15,4:16]=True;ridges[4:16,4]=True;ridges[4:16,15]=True
        self.assertEqual(_segments(ridges)[0],[])

    def test_new_path_never_crosses_a_third_original_ridge(self):
        image,observed,ridges,metadata=junction_fixture()
        ridges[46,30:100]=True
        observed=soften_ridges(ridges,np.full(image.shape,.8,np.float32),.5)
        old,metadata=completion_comparison(image,observed,trace_ridges=ridges,settings=SETTINGS)
        updated,_,_=fragment_completion(image,observed,old['trace_ridges'],metadata)
        for c in updated['candidates'][len(metadata['candidates']):]:
            path=np.asarray(c['path_yx'])[1:-1]
            self.assertFalse(old['trace_ridges'][tuple(path.T)].any())

    def test_each_selected_connection_uses_a_physical_anchor_site_once(self):
        image,observed,ridges,metadata=junction_fixture()
        updated,_,_=fragment_completion(image,observed,ridges,metadata)
        sites=[tuple(c['path_yx'][i]) for c in updated['candidates'] if c['geometry_selected'] for i in (0,-1)]
        self.assertEqual(len(sites),len(set(sites)))

    def test_candidate_gap_length_and_coordinate_bounds_are_recorded_and_finite(self):
        image,observed,ridges,metadata=junction_fixture()
        updated,probe,_=fragment_completion(image,observed,ridges,metadata)
        for c in updated['candidates'][len(metadata['candidates']):]:
            self.assertLessEqual(c['gap'],metadata['gap_limit_pixels']+1e-8)
            self.assertLessEqual(c['length'],SETTINGS.max_length_ratio*c['gap']+1e-8)
            self.assertTrue(0<=c['region_disagreement']<=1)
        json.dumps(probe,allow_nan=False)

    def test_region_descriptor_uses_travel_direction_and_remains_soft(self):
        a=dict(side_means=[.2,.7],context_valid=True)
        same=dict(side_means=[.7,.2],context_valid=True)
        reversed_regions=dict(side_means=[.2,.7],context_valid=True)
        self.assertEqual(_region_cost(a,same,.01),0)
        self.assertGreater(_region_cost(a,reversed_regions,.01),0)
        self.assertLessEqual(_region_cost(a,reversed_regions,.01),1)
        self.assertEqual(_region_cost(dict(a,context_valid=False),same,.01),0)

    def test_polarity_reversal_does_not_impose_a_visible_light_color_rule(self):
        image,observed,ridges,metadata=junction_fixture()
        a,_,_=fragment_completion(image,observed,ridges,metadata)
        _,inverted_meta=completion_comparison(1-image,observed,trace_ridges=ridges,settings=SETTINGS)
        b,_,_=fragment_completion(1-image,observed,ridges,inverted_meta)
        paths=lambda m:{tuple(tuple(p) for p in c['path_yx']) for c in m['candidates'] if c['reliability_accepted']}
        self.assertEqual(paths(a),paths(b))

    def test_rotation_does_not_depend_on_the_image_horizon(self):
        image,observed,ridges,metadata=junction_fixture()
        a,_,_=fragment_completion(image,observed,ridges,metadata)
        im,obs,r=[np.rot90(x).copy() for x in (image,observed,ridges)]
        old,meta=completion_comparison(im,obs,trace_ridges=r,settings=SETTINGS)
        b,_,_=fragment_completion(im,obs,old['trace_ridges'],meta)
        self.assertGreater(a['fragment_completion']['new_reliability_accepted'],0)
        self.assertGreater(b['fragment_completion']['new_reliability_accepted'],0)

    def test_anchor_and_search_limits_are_explicit_not_silent(self):
        image,observed,ridges,metadata=junction_fixture()
        updated,_,_=fragment_completion(image,observed,ridges,metadata,settings=FragmentSettings(max_anchors=1,max_searches=1))
        self.assertGreater(updated['fragment_completion']['truncated_anchors'],0)
        self.assertLessEqual(updated['fragment_completion']['new_candidates'],1)
        a,_,_=fragment_completion(image,observed,ridges,metadata,settings=FragmentSettings(max_searches=1))
        b,_,_=fragment_completion(image,observed,ridges,metadata,settings=FragmentSettings(max_searches=1))
        self.assertEqual(a,b)
        self.assertGreater(a['fragment_completion']['truncated_pairs'],0)

    def test_invalid_inputs_fail_before_search_and_duplicate_application_is_refused(self):
        for options in (dict(max_searches=True),dict(max_anchors=1.5),dict(minimum_tangent_quality=float('nan')),dict(minimum_fragment_fraction=.2)):
            with self.subTest(options=options),self.assertRaises(ValueError):FragmentSettings(**options)
        image,observed,ridges,metadata=junction_fixture()
        with self.assertRaises(ValueError):fragment_completion(image,observed,ridges.astype(np.float32),metadata)
        with self.assertRaises(TypeError):fragment_completion(image,observed,ridges,metadata,settings={})
        updated,_,_=fragment_completion(image,observed,ridges,metadata)
        with self.assertRaises(ValueError):fragment_completion(image,observed,ridges,updated)

    def test_backend_anchor_revisit_is_rejected_before_confidence_and_matching(self):
        import models.fragment_completion as module
        original_search=module._find_path
        def looping(*args,**kwargs):
            result,reason,expanded=original_search(*args,**kwargs)
            if result is None:return result,reason,expanded
            path,cost,length=result
            path=np.concatenate((path[:2],path[:1],path[1:]),axis=0)
            return (path,cost,length+2),reason,expanded
        image,E,ridges,metadata=junction_fixture()
        with patch.object(module,'_find_path',side_effect=looping):
            updated,_,_=fragment_completion(image,E,ridges,metadata)
        self.assertEqual(updated['fragment_completion']['new_candidates'],0)
        self.assertGreater(updated['fragment_completion']['pair_rejections'].get('non_simple_path',0),0)
        self.assertEqual(updated['candidates'],metadata['candidates'])

    def test_cached_pool_replay_discards_loops_and_matches_fresh_valid_pool(self):
        image,E,ridges,metadata=junction_fixture()
        structure=np.full_like(image,.5);clutter=np.zeros_like(image)
        _,base=prioritize_completion(E,metadata,structure,clutter)
        raw,_,_=fragment_completion(image,E,ridges,base)
        _,fresh=prioritize_completion(E,raw,structure,clutter)
        cached=copy.deepcopy(fresh);invalid=copy.deepcopy(cached['candidates'][-1])
        path=np.asarray(invalid['path_yx']);invalid['path_yx']=np.concatenate((path[:2],path[:1],path[1:])).tolist()
        invalid['id']=len(cached['candidates']);cached['candidates'].append(invalid)
        frozen=json.dumps(cached,sort_keys=True)
        clean=sanitize_cached_fragments(image,E,ridges,base,cached)
        self.assertEqual(clean['candidates'],fresh['candidates'])
        self.assertEqual(clean['fragment_completion']['cached_search_replay']['rejection_counts'],{'non_simple_path':1})
        self.assertEqual(frozen,json.dumps(cached,sort_keys=True))


class RematchTests(unittest.TestCase):
    def test_backup_can_use_an_endpoint_released_by_a_rejected_connection(self):
        image,observed,ridges,metadata=paired_fixture()
        self.assertEqual(len(metadata['candidates']),1)
        # A constructed decoding counterexample, not a claim that the original
        # ambiguity formula produces this configuration on real thermal data.
        rejected=metadata['candidates'][0]
        rejected.update(reliability_score=.2,reliability_accepted=False,reliability_rejections=['low_score'])
        backup=copy.deepcopy(rejected);backup.update(id=1,cost=rejected['cost']+.5,
            reliability_score=.8,reliability_rejections=[],geometry_selected=False,geometry_rejection='endpoint_already_used')
        metadata['candidates'].append(backup)
        before=json.dumps(metadata,sort_keys=True)
        maps,updated=rematch_fixed_C(image,observed,ridges,metadata)
        self.assertEqual(updated['rematch']['new_accepted_ids'],[1])
        self.assertEqual(updated['candidates'][0]['reliability_score'],.2)
        self.assertEqual(updated['candidates'][1]['reliability_score'],.8)
        self.assertEqual(before,json.dumps(metadata,sort_keys=True))
        self.assertTrue((maps['rematched']>=maps['baseline']).all())

    def test_existing_accepted_paths_C_and_R_survive_rematch(self):
        image,observed,ridges,metadata=paired_fixture()
        _,updated=rematch_fixed_C(image,observed,ridges,metadata)
        for a,b in zip(metadata['candidates'],updated['candidates']):
            for key in ('features','reliability_score','reliability_rejections','path_yx','strength'):self.assertEqual(a[key],b[key])
            if a['reliability_accepted']:self.assertEqual(a,b)
        base,base_meta=prioritize_completion(observed,metadata,np.ones_like(image),np.zeros_like(image))
        new,new_meta=prioritize_completion(observed,updated,np.ones_like(image),np.zeros_like(image))
        self.assertTrue((new['priority']>=base['priority']).all())
        self.assertEqual([c['add_importance'] for c in base_meta['candidates']],[c['add_importance'] for c in new_meta['candidates']])

    def test_rematch_does_not_rescore_an_ambiguity_rejected_backup(self):
        image,observed,ridges,metadata=paired_fixture()
        original=metadata['candidates'][0]
        original.update(reliability_accepted=False,reliability_score=.4,reliability_rejections=['ambiguous_match'])
        _,updated=rematch_fixed_C(image,observed,ridges,metadata)
        self.assertEqual(updated['rematch']['new_accepted_ids'],[])
        self.assertEqual(updated['candidates'][0]['features'],original['features'])


class PreviewTests(unittest.TestCase):
    def test_cached_R_renderer_matches_fresh_conditions_and_control_gain(self):
        from replay_fragment_safety_fix import render_cached_R
        image,E,ridges,metadata=junction_fixture();gray=np.rint(image*255).astype(np.uint8)
        old,metadata=completion_comparison(preview._image01(gray),E,trace_ridges=ridges,settings=SETTINGS)
        data=(gray,E,old['reliable'],metadata,np.full_like(image,.5),np.zeros_like(image),None,None,old['trace_ridges'])
        maps,records=preview.apply_arms(data,FragmentSettings())
        for arm in ('fragment','observation'):
            cached=copy.deepcopy(records[arm]);actual=render_cached_R(E,cached)
            np.testing.assert_array_equal(actual,maps[arm])
            self.assertEqual(cached['importance']['uniform_mean_control_gain'],records[arm]['importance']['uniform_mean_control_gain'])

    def test_cli_refuses_overwrite_and_test_split_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):preview.parse_args(['--output-dir',tmp])
        for args in (['--external-samples','25'],['--erasure-samples','41'],['--zoom','0'],['--input-dir','FLIR_datasets/testB'],['--max-searches','0']):
            with self.assertRaises(SystemExit):preview.parse_args(args)

    def test_arm_comparison_keeps_original_and_separates_encoding_from_matching(self):
        image,E,ridges,metadata=junction_fixture()
        gray=np.rint(image*255).astype(np.uint8);image=preview._image01(gray)
        old,metadata=completion_comparison(image,E,trace_ridges=ridges,settings=SETTINGS)
        data=(gray,E,old['reliable'],metadata,np.full_like(image,.5),np.zeros_like(image),None,None,old['trace_ridges'])
        maps,records=preview.apply_arms(data,FragmentSettings())
        self.assertEqual(records['rematch']['rematch']['C_rescored'],False)
        self.assertTrue((maps['fragment']>=maps['baseline']).all())
        self.assertTrue((maps['observation']>=maps['baseline']).all())
        self.assertTrue(maps['fragment_mask'].any())
        for a,b in zip(records['fragment']['candidates'],records['observation']['candidates']):
            for key in ('path_yx','reliability_score','add_importance','importance_gain'):self.assertEqual(a[key],b[key])

    def test_real_writer_exports_all_controls_without_overwriting_condition_images(self):
        from PIL import Image
        image,E,ridges,metadata=junction_fixture();gray=np.rint(image*255).astype(np.uint8);image=preview._image01(gray)
        old,metadata=completion_comparison(image,E,trace_ridges=ridges,settings=SETTINGS)
        data=(gray,E,old['reliable'],metadata,np.full_like(image,.5),np.zeros_like(image),None,None,old['trace_ridges'])
        base,_=prioritize_completion(E,metadata,data[4],data[5])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);imp=root/'importance';weak=root/'weak'
            for folder in (imp,weak):(folder/'maps').mkdir(parents=True)
            np.savez(imp/'maps/01_importance.npz',priority=base['priority'])
            np.savez(weak/'maps/01_recall.npz',observed=E,baseline=base['priority'],recall=base['priority'])
            row=dict(prefix='01',filename='fixture.png',dataset='FLIR-development',video_id='fixture',semantic_mode='none')
            args=type('Args',(),dict(output_dir=root/'output',importance_dir=imp,weak_dir=weak,external_samples=0,
                erasure_samples=0,annotations=None,zoom=2,max_anchors=8192,max_searches=512))()
            identity=dict(maps_sha256={'01':sha256_file(imp/'maps/01_importance.npz')})
            weak_identity=({'plan':{'code_sha256':{},'controls':[]}},dict(map_sha256={'01':sha256_file(weak/'maps/01_recall.npz')}))
            with patch.object(preview,'preflight',return_value=(dict(targets=[]),[row])),\
                 patch.object(preview,'select_external_training',return_value=([],{})),\
                 patch.object(preview,'source_identity',return_value=identity),\
                 patch.object(preview,'weak_identity',return_value=weak_identity),\
                 patch.object(preview,'control_plan',return_value=([],{},[])),\
                 patch.object(preview,'verify_source_files'),\
                 patch.object(preview,'load_input',return_value=data):
                preview.run(args)
            status=json.loads((args.output_dir/'status.json').read_text())
            self.assertEqual(status['phase'],'completed')
            self.assertFalse(status['real_gap_quality_verified'])
            for filename in ('01_comparison.png','01_rematch.png','01_amplitude.png'):
                with Image.open(args.output_dir/'frames'/filename) as im:self.assertGreater(im.width,gray.shape[1])
            with Image.open(args.output_dir/'frames/01_rematch_condition.png') as im:
                self.assertEqual(im.size,(gray.shape[1],gray.shape[0]));self.assertEqual(im.mode,'L')
            with np.load(args.output_dir/'maps/01_fragments.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['baseline'],base['priority']);self.assertTrue((z['fragment']>=z['baseline']).all())
            self.assertEqual((args.output_dir/'gap_statistics.csv').read_text(encoding='utf-8-sig').strip(),'id,arm,label')
            self.assertTrue((args.output_dir/'annotate_gaps.html').is_file())


if __name__=='__main__':unittest.main()
