"""Checks for the separate regional attention audit, not a training variant."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from region_attention_probe import match_mean, region_attention
from audit_c2_importance import fuse_fixed_candidates


class RegionAttentionTests(unittest.TestCase):
    def test_flat_regions_do_not_create_edges(self):
        for value in (0.,.5,1.):
            image = np.full((32,40),value,np.float32)
            importance,cues = region_attention(image,np.full_like(image,.32))
            np.testing.assert_array_equal(cues["region_score"],np.zeros_like(image))
            self.assertLess(float(importance.max()),.32)
            edge,_ = fuse_fixed_candidates(np.zeros_like(image),np.zeros_like(image),importance,np.zeros_like(image))
            self.assertFalse(edge.any())

    def test_region_interior_is_not_inserted_as_edge(self):
        image = np.full((64,80),.3,np.float32)
        image[16:48,30:45] = .7
        old = np.full_like(image,.32)
        initial = image.copy()
        importance,cues = region_attention(image,old)
        self.assertGreater(float(cues["region_score"][25:39,33:42].mean()),.2)
        self.assertGreater(float(importance.max()),.32)
        # A contrast score need not distinguish which of two adjacent regions
        # is the foreground. Do not encode semantic selectivity as a unit-test
        # assumption; the real-image audit must measure it.
        fine = np.zeros_like(image)
        fine[16:48,30] = 1.
        fine[16:48,44] = 1.
        edge,_ = fuse_fixed_candidates(fine,fine,importance,np.zeros_like(image))
        self.assertEqual(float(edge[20:44,33:42].max()),0.)
        self.assertTrue(np.all(edge>=.25*fine))
        np.testing.assert_array_equal(image,initial)
        np.testing.assert_array_equal(old,np.full_like(image,.32))
        second,again = region_attention(image,old)
        np.testing.assert_array_equal(importance,second)
        np.testing.assert_array_equal(cues["labels"],again["labels"])

    def test_mean_control_handles_dimming_brightening_and_saturation(self):
        baseline = np.linspace(0,1,100,dtype=np.float32).reshape(10,10)
        for target in (baseline*.4,np.minimum(baseline*2,1),np.zeros_like(baseline)):
            control,gain = match_mean(baseline,target)
            self.assertAlmostEqual(float(control.mean()),float(target.mean()),places=6)
            np.testing.assert_allclose(control,np.clip(baseline*gain,0,1),atol=1e-7)
        with self.assertRaises(ValueError):
            match_mean(np.zeros_like(baseline),np.ones_like(baseline))

    def test_invalid_inputs_rejected(self):
        good = np.zeros((16,16),np.float32)
        for bad in (np.full_like(good,np.nan),np.full_like(good,1.1),np.zeros((16,)),np.zeros((0,0))):
            with self.assertRaises(ValueError):
                region_attention(bad,good)
        with self.assertRaises(ValueError):
            region_attention(good,np.zeros((16,15),np.float32))


if __name__ == "__main__":
    unittest.main()
