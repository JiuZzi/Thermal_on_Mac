"""Mechanism and interface checks; these do not establish contour quality."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"pytorch-CycleGAN-and-pix2pix"))

from models.semantic_edges import (
    FMB_CLASS_NAMES, semantic_importance, fuse_candidates, make_semantic_contours,
    structure_semantic_importance, make_structure_semantic_contours,
    make_structure_semantic_condition,
)
from models.conditioned_generator import ConditionedGenerator
from preview_semantic_contours import read_cache, make_sheet, parse_args
from preview_structure_semantic_contours import match_global_mean


def scores(category,shape):
    values=np.zeros((15,*shape),np.float32)
    values[FMB_CLASS_NAMES.index(category)]=1
    return values


class SemanticContourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_semantic_roles_and_bounded_responses_preserve_both_routes(self):
        for category in ("person","car","bicycle","building"):
            importance,support=semantic_importance(scores(category,(5,7)))
            np.testing.assert_allclose(importance,.85,atol=1e-7)
            np.testing.assert_array_equal(support,1)
        importance,support=semantic_importance(scores("vegetation",(5,7)))
        np.testing.assert_allclose(importance,.18,atol=1e-7)
        mixed=.3*scores("person",(5,7))+.7*scores("vegetation",(5,7))
        importance,support=semantic_importance(mixed)
        self.assertTrue(np.all((importance>.18)&(importance<.85)))
        np.testing.assert_allclose(support,.3,atol=1e-7)

    def test_taxonomy_names_prevent_channel_index_mixups(self):
        original=scores("person",(6,8))
        permutation=np.arange(15)[::-1]
        names=tuple(FMB_CLASS_NAMES[i] for i in permutation)
        a=semantic_importance(original)[0]
        b=semantic_importance(original[permutation],names)[0]
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):
            semantic_importance(original,tuple(["person"]*15))

    def test_rejects_invalid_scores_and_coordinate_mismatch(self):
        original=scores("person",(6,8))
        for invalid in (original[:14],original*.9,original+float("nan"),original-1):
            with self.assertRaises(ValueError):
                semantic_importance(invalid)
        with self.assertRaisesRegex(ValueError,"coordinates"):
            make_semantic_contours(np.zeros((8,6),np.float32),original)

    def test_wrong_semantics_preserve_fine_floor_and_coarse_route(self):
        fine=np.ones((5,7),np.float32);zero=np.zeros_like(fine)
        low=semantic_importance(scores("vegetation",fine.shape))[0]
        high=semantic_importance(scores("person",fine.shape))[0]
        self.assertTrue(np.all(fuse_candidates(fine,zero,fine,low)>=.25*fine))
        self.assertTrue(np.all(fuse_candidates(zero,fine,fine,high)>0))
        np.testing.assert_array_equal(fuse_candidates(zero,zero,zero,high),zero)

    def test_flat_thermal_input_does_not_invent_semantic_boundary_edges(self):
        image=np.full((48,61),.5,np.float32)
        probabilities=scores("vegetation",image.shape)
        probabilities[:,15:35,21:30]=scores("person",(20,9))
        result=make_semantic_contours(image,probabilities)
        for mode in ("c2b","c2c","semantic512"):
            np.testing.assert_array_equal(result[mode],np.zeros_like(image))

    def test_existing_controls_match_and_direct_input_keeps_confidence_zero(self):
        gray=np.zeros((48,61),np.uint8);gray[12:36,25:33]=230
        tir=torch.from_numpy((gray.astype(np.float32)/255-.5)/.5)[None,None]
        image=(tir.numpy()[0,0]+1)/2
        result=make_semantic_contours(image,scores("person",image.shape))
        for mode,condition in (("c2b","soft_multi"),("c2c","soft_saliency")):
            model=ConditionedGenerator(nn.Identity(),condition,fusion_mode="direct")
            edge,reserved=model.make_condition(tir)
            np.testing.assert_array_equal(edge[0,0].numpy(),result[mode])
            np.testing.assert_array_equal(reserved.numpy(),0)
        model=ConditionedGenerator(nn.Identity(),"soft_saliency",fusion_mode="direct")
        output=model(tir,edge_override=torch.from_numpy(result["semantic512"])[None,None])
        np.testing.assert_array_equal(output[:,0:1].numpy(),tir.numpy())
        np.testing.assert_array_equal(output[0,1].numpy(),result["semantic512"])
        np.testing.assert_array_equal(output[:,2:3].numpy(),0)

    def test_cache_rejects_wrong_gray_and_uses_base_not_detail(self):
        gray=np.zeros((6,8),np.uint8);prob=scores("person",gray.shape)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"maps.npz"
            np.savez(path,TIR_uint8=gray,class_names=FMB_CLASS_NAMES,base_probabilities=prob,
                     base_labels=prob.argmax(0).astype(np.uint8),detail_probabilities=scores("vegetation",gray.shape))
            _,actual=read_cache(path,gray)
            np.testing.assert_array_equal(actual,prob)
            with self.assertRaises(ValueError):
                read_cache(path,gray+1)

    def test_visual_scale_and_cli_overwrite_protection(self):
        gray=np.zeros((10,12),np.uint8)
        maps={key:np.full(gray.shape,.25,np.float32) for key in ("c2b","c2c","semantic512","semantic_importance")}
        with tempfile.TemporaryDirectory() as directory:
            sheet=make_sheet(gray,maps,crop=[2,3,8,9],box=[3,4,7,8],zoom=3)
            self.assertEqual(sheet.size,(750,92))
            array=np.asarray(sheet)
            self.assertTrue(np.any(np.all(array==[64,64,64],axis=-1)))
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(["--output-dir",directory])


class StructureSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_condition_matches_review_for_mixed_scores_and_geometries(self):
        rng = np.random.default_rng(42)
        for shape in ((32, 40), (64, 64), (256, 256)):
            image = rng.random(shape, dtype=np.float32)
            probabilities = rng.random((15, *shape), dtype=np.float32)
            probabilities /= probabilities.sum(0, keepdims=True)
            for gray, responses in ((image, probabilities),
                                    (image[:, ::-1], probabilities[:, :, ::-1]),
                                    (np.full(shape, .5, np.float32), scores("person", shape))):
                with self.subTest(shape=shape):
                    expected = make_structure_semantic_contours(gray, responses)["joint512"]
                    actual = make_structure_semantic_condition(gray, responses)
                    np.testing.assert_array_equal(actual, expected)

    def test_training_condition_keeps_input_validation(self):
        image = np.zeros((32, 40), np.float32)
        with self.assertRaisesRegex(ValueError, "coordinates"):
            make_structure_semantic_condition(image, scores("person", (40, 32)))
        for invalid in (scores("person", image.shape) * .9,
                        scores("person", image.shape) + float("nan")):
            with self.assertRaises(ValueError):
                make_structure_semantic_condition(image, invalid)

    def test_clear_fine_structure_can_survive_wrong_background_semantics(self):
        shape=(5,7);ones=np.ones(shape,np.float32);zero=np.zeros(shape,np.float32)
        base=ones*.7;semantic=ones*.18
        joint,gate,_=structure_semantic_importance(base,semantic,zero,ones,zero,zero)
        np.testing.assert_allclose(joint,base,atol=1e-6)
        self.assertTrue(np.all(gate>.99))
        self.assertTrue(np.all(fuse_candidates(ones,zero,zero,joint)>fuse_candidates(ones,zero,zero,semantic)))

    def test_dense_texture_and_coarse_only_routes_close_structural_rescue(self):
        shape=(5,7);ones=np.ones(shape,np.float32);zero=np.zeros(shape,np.float32)
        base=ones*.7;semantic=ones*.18
        for fine,coarse,clutter in ((ones,zero,ones),(zero,ones,zero)):
            joint,gate,_=structure_semantic_importance(base,semantic,zero,fine,coarse,clutter)
            np.testing.assert_array_equal(joint,semantic)
            np.testing.assert_array_equal(gate,zero)
            np.testing.assert_array_equal(fuse_candidates(fine,coarse,clutter,joint),fuse_candidates(fine,coarse,clutter,semantic))

    def test_strong_semantic_route_is_preserved_without_unconditional_averaging(self):
        shape=(5,7);ones=np.ones(shape,np.float32);zero=np.zeros(shape,np.float32)
        base=ones*.6;semantic=ones*.85
        joint,gate,_=structure_semantic_importance(base,semantic,ones,ones,zero,zero)
        np.testing.assert_array_equal(joint,semantic)
        np.testing.assert_array_equal(gate,zero)

    def test_random_maps_preserve_range_semantic_edges_and_fine_floor(self):
        rng=np.random.default_rng(42);shape=(19,23)
        support=rng.random(shape,dtype=np.float32)
        semantic=.18+.67*support;base=.18+.67*rng.random(shape,dtype=np.float32)
        fine,coarse,clutter=[rng.random(shape,dtype=np.float32) for _ in range(3)]
        joint,gate,_=structure_semantic_importance(base,semantic,support,fine,coarse,clutter)
        self.assertTrue(np.all(joint>=semantic-1e-7))
        self.assertTrue(np.all(joint<=np.maximum(base,semantic)+1e-7))
        old=fuse_candidates(fine,coarse,clutter,semantic);new=fuse_candidates(fine,coarse,clutter,joint)
        self.assertTrue(np.all(new+2e-7>=old))
        self.assertTrue(np.all(new+1e-7>=.25*fine))
        np.testing.assert_array_equal(joint,structure_semantic_importance(base[:,::-1],semantic[:,::-1],support[:,::-1],fine[:,::-1],coarse[:,::-1],clutter[:,::-1])[0][:,::-1])

    def test_rejects_misaligned_or_out_of_range_structure_priors(self):
        shape=(5,7);ones=np.ones(shape,np.float32);zero=np.zeros(shape,np.float32)
        for base in (ones*.1,ones*.9,np.ones((7,5),np.float32)*.6):
            with self.assertRaises(ValueError):
                structure_semantic_importance(base,ones*.18,zero,ones,zero,zero)

    def test_joint_helper_does_not_create_edges_from_a_semantic_region_on_flat_tir(self):
        image=np.full((48,61),.5,np.float32);probabilities=scores('vegetation',image.shape)
        probabilities[:,10:30,22:31]=scores('person',(20,9))
        old=make_semantic_contours(image,probabilities)
        new=make_structure_semantic_contours(image,probabilities)
        for key,value in old.items():np.testing.assert_array_equal(new[key],value)
        np.testing.assert_array_equal(new['joint512'],np.zeros_like(image))

    def test_brightness_control_matches_global_mean_without_target_information(self):
        original=np.array([[0,.1,.3],[.5,.8,1]],np.float32)
        reference=original.copy();reference[0,1]+=.2;reference[1,0]+=.1
        matched,gain=match_global_mean(original,reference)
        self.assertGreater(gain,1)
        self.assertLess(abs(float(matched.mean(dtype=np.float64)-reference.mean(dtype=np.float64))),1e-7)
        self.assertEqual(matched[0,0],0)
        empty,gain=match_global_mean(np.zeros_like(original),np.zeros_like(original))
        np.testing.assert_array_equal(empty,0)


if __name__=="__main__":
    unittest.main()
