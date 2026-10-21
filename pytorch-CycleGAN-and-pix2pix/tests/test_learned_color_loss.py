"""Loss accepts multiple modes, preserves Q gating and backpropagates RGB only."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from skimage.color import lab2rgb
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.learned_color_prior import LearnedChromaPrior


def bank_file(root,second=False):
    support=dict(source='group_balanced_train_only',quantile=.99,max_nll=10.)
    item=dict(status='supported',weights=[.99,.01],means_ab=[[-20,25],[15,25]],
              variances_ab=[[4,9],[9,4]],loss_support=support)
    sky=dict(status='supported',weights=[1.],means_ab=[[10,-30]],variances_ab=[[9,9]],loss_support=support) if second else dict(status='unsupported')
    path=root/'bank.json'
    path.write_text(json.dumps(dict(format='learned_chroma_gmm_v1',class_names=['vegetation','sky'],
                                    classes=dict(vegetation=item,sky=sky))))
    return path


def image(ab):
    rgb=lab2rgb(np.asarray([[[65.,*ab]]]))[0,0]
    return torch.tensor(rgb,dtype=torch.float32)[None,:,None,None].expand(1,3,4,8).clone()*2-1


def scores():
    p=torch.zeros(1,3,4,8); p[:,0]=1
    return p,torch.ones(1,1,4,8)


class LearnedColorLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=bank_file(Path(self.tmp.name))
        self.prior=LearnedChromaPrior(self.path,['vegetation','sky','other'],require_loss_support=True)

    def test_common_and_rare_modes_are_accepted_but_average_is_not_a_target(self):
        p,q=scores()
        for ab in ([-20,25],[15,25]):
            loss,coverage=self.prior.chroma_loss(image(ab),p,q)
            self.assertEqual(float(loss),0)
            self.assertEqual(float(coverage),1)
        outside,_=self.prior.chroma_loss(image([-2.5,25]),p,q)
        self.assertGreater(float(outside),.8)
        self.assertLess(float(outside),1)

    def test_rgb_has_finite_gradient_while_semantics_and_quality_stay_frozen(self):
        p,q=scores(); p.requires_grad_(); q.requires_grad_()
        rgb=image([-10,25]).requires_grad_()
        loss,_=self.prior.chroma_loss(rgb,p,q)
        loss.backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertGreater(float(rgb.grad.abs().sum()),0)
        self.assertIsNone(p.grad); self.assertIsNone(q.grad)
        self.assertFalse(any(x.requires_grad for x in self.prior.buffers()))

    def test_quality_is_not_cancelled_by_normalization_and_unreliable_or_unsupported_pixels_are_off(self):
        p,q=scores(); rgb=image([-10,25]).requires_grad_()
        full,_=self.prior.chroma_loss(rgb,p,q,min_quality=.1)
        half,_=self.prior.chroma_loss(rgb,p,q*.5,min_quality=.1)
        torch.testing.assert_close(half,full*.5)
        cases=[(p,q*.49),(p,q*0),(p.roll(1,1),q),(p.roll(2,1),q)]
        for probabilities,quality in cases:
            loss,coverage=self.prior.chroma_loss(rgb,probabilities,quality)
            self.assertEqual(float(loss),0); self.assertEqual(float(coverage),0)
            loss.backward()
        self.assertEqual(rgb.grad.count_nonzero(),0)

    def test_equal_class_weight_despite_unequal_region_areas(self):
        path=bank_file(Path(self.tmp.name),second=True)
        prior=LearnedChromaPrior(path,['vegetation','sky','other'],require_loss_support=True)
        p,q=scores(); p[:,:,:,2:]=0; p[:,1,:,2:]=1
        rgb=image([-10,25])
        rgb[:,:,:,2:]=image([10,-10])[:,:,:,2:]
        together,_=prior.chroma_loss(rgb,p,q)
        first,_=prior.chroma_loss(rgb[:,:,:,:2],p[:,:,:,:2],q[:,:,:,:2])
        second,_=prior.chroma_loss(rgb[:,:,:,2:],p[:,:,:,2:],q[:,:,:,2:])
        torch.testing.assert_close(together,(first+second)/2)

    def test_old_bank_is_usable_without_loss_but_enabling_loss_requires_refit(self):
        bank=json.loads(self.path.read_text()); del bank['classes']['vegetation']['loss_support']
        self.path.write_text(json.dumps(bank))
        LearnedChromaPrior(self.path,['vegetation','sky','other'])
        with self.assertRaisesRegex(ValueError,'refit'):
            LearnedChromaPrior(self.path,['vegetation','sky','other'],require_loss_support=True)

    def test_loss_rejects_invalid_quality_or_geometry(self):
        p,q=scores(); rgb=image([-10,25])
        for threshold in (0,-.1,1.01,float('nan')):
            with self.assertRaises(ValueError): self.prior.chroma_loss(rgb,p,q,threshold)
        with self.assertRaises(ValueError): self.prior.chroma_loss(rgb,p,q*2)
        with self.assertRaises(ValueError): self.prior.chroma_loss(rgb,p[:,:,:,:2],q)


if __name__=='__main__': unittest.main()
