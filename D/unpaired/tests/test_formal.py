import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from D.tests import test_pipeline as tiny
from D.unpaired.model import FormalGenerator, convert_edge_encoder
from D.unpaired.protocol import config_from, IndexStream, ReplayPool, validate_arrays
from D.unpaired.losses import edge_preservation, spatial_content, self_similarity, semantic_consistency, chroma_smoothness


def bank(path):
    path.write_text(json.dumps(dict(format="learned_chroma_gmm_v1", class_names=["road"], classes={
        "road":dict(status="supported",weights=[1.], means_ab=[[0.,0.]],variances_ab=[[100.,100.]],
                    loss_support=dict(source="group_balanced_train_only",quantile=.99,max_nll=15.))})))
    return path


def conditions(size=64):
    gray = torch.full((1,1,size,size), .3)
    gray[:,:,14:52,20:45] = .7
    e = torch.zeros_like(gray); e[:,:,14:52,20] = .7; e[:,:,14:52,44] = .7
    p = torch.zeros((1,15,size,size)); p[:,1] = 1
    return torch.cat((gray,e,torch.zeros_like(e)),1), p, torch.ones_like(gray)


class LossAndProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_edge_support_prefers_an_aligned_boundary_and_preserves_gradient(self):
        x,p,q = conditions()
        aligned = (x[:,:1].repeat(1,3,1,1)*2-1).requires_grad_()
        shifted = aligned.detach().roll(12,-1).requires_grad_()
        good = edge_preservation(aligned,x[:,:1],x[:,1:2])
        bad = edge_preservation(shifted,x[:,:1],x[:,1:2])
        self.assertLess(float(good),float(bad))
        bad.backward(); self.assertGreater(float(shifted.grad.abs().sum()),0)

    def test_self_similarity_allows_polarity_change_and_detects_spatial_mismatch(self):
        x,p,q = conditions()
        np.testing.assert_allclose(self_similarity(x[:,:1]).numpy(),self_similarity(1-x[:,:1]).numpy(),atol=1e-6)
        good = x[:,:1].repeat(1,3,1,1)*2-1
        bad = good.roll(11,-1).requires_grad_()
        normal = spatial_content(good,x[:,:1],x[:,1:2])
        wrong = spatial_content(bad,x[:,:1],x[:,1:2])
        self.assertLess(float(normal),float(wrong))
        wrong.backward(); self.assertTrue(torch.isfinite(bad.grad).all())

    def test_flat_and_saturated_regions_have_finite_edge_gradients(self):
        source = torch.linspace(0,1,64)[None,None,None,:].expand(1,1,64,64)
        edge = torch.zeros_like(source); edge[:,:,:,30:33] = .8
        for value in (-1., 0., 1.):
            for support in (edge, edge*0):
                with self.subTest(value=value, empty=not bool(support.any())):
                    generated = torch.full((1,3,64,64), value, requires_grad=True)
                    loss = edge_preservation(generated,source,support)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(generated.grad).all())
                    if not support.any():
                        self.assertEqual(float(loss),0.)
                        self.assertEqual(float(generated.grad.abs().sum()),0.)

    def test_semantic_loss_empty_mask_is_zero_and_color_regularizer_is_finite(self):
        x,p,q = conditions()
        logits = torch.zeros_like(p, requires_grad=True)
        loss = semantic_consistency(logits.softmax(1),p,q*0)
        self.assertEqual(float(loss),0)
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        rgb = (x[:,:1].repeat(1,3,1,1)*2-1).requires_grad_()
        chroma_smoothness(rgb,x[:,1:2],p).backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())

    def test_domain_streams_and_replay_restore_without_pairing_domains(self):
        tir, rgb = IndexStream(20,"TIR"), IndexStream(20,"RGB")
        self.assertNotEqual([tir.next() for _ in range(20)],[rgb.next() for _ in range(20)])
        state = tir.state(); expected = [tir.next() for _ in range(25)]
        resumed = IndexStream(20,"TIR"); resumed.restore(state)
        self.assertEqual(expected,[resumed.next() for _ in range(25)])
        pool = ReplayPool(2)
        for n in range(3): pool.query(torch.full((1,1,2,2),float(n)))
        saved = copy.deepcopy(pool.state()); expected = pool.query(torch.ones(1,1,2,2)*7)
        restored = ReplayPool(2); restored.restore(saved)
        self.assertTrue(torch.equal(expected,restored.query(torch.ones(1,1,2,2)*7)))

    def test_cache_validation_rejects_nonfinite_and_wrong_probability_mass(self):
        gray = np.zeros((8,8),np.uint8); edge = np.zeros((8,8),np.float32)
        p = np.zeros((15,8,8),np.float32); p[1] = 1
        validate_arrays(gray,edge,p,edge)
        with self.assertRaises(ValueError): validate_arrays(gray,edge,p*.5,edge)
        broken = edge.copy(); broken[0,0] = np.nan
        with self.assertRaises(ValueError): validate_arrays(gray,broken,p,edge)


class FormalGeneratorTests(unittest.TestCase):
    def setUp(self):
        tiny.TinyPipelineTests.setUpClass()
        self.temp = tempfile.TemporaryDirectory()
        self.path = bank(Path(self.temp.name)/"prior.json")
        self.config = config_from(); self.config.update(width=64,height=64,sampling_steps=4,strength=.75,lora_rank=2)
        self.model = FormalGenerator(tiny.TinyPipelineTests.pipeline,self.config,self.path,{"fixture":"v1"})

    def tearDown(self):
        self.temp.cleanup(); tiny.TinyPipelineTests.tearDownClass()

    def test_edge_weight_conversion_preserves_the_original_encoder_response(self):
        control = tiny.TinyPipelineTests.plain.unet  # Use a fresh standalone convolution container.
        del control
        from types import SimpleNamespace
        conv = nn.Conv2d(3,5,3,padding=1)
        edge = torch.rand(1,1,16,16)
        expected = conv(edge.repeat(1,3,1,1)).detach()
        wrapper = SimpleNamespace(controlnet_cond_embedding=SimpleNamespace(conv_in=conv))
        convert_edge_encoder(wrapper)
        actual = conv(torch.cat((torch.rand_like(edge),edge,torch.zeros_like(edge)),1))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_full_sampler_has_tir_gradient_zero_confidence_gradient_and_frozen_base(self):
        x,p,q = conditions()
        frozen = self.model.unet.conv_in.weight.detach().clone()
        y = self.model(x,p,q)
        self.assertEqual(y.shape,(1,3,64,64))
        (y.square().mean()+edge_preservation(y,x[:,:1],x[:,1:2])).backward()
        grad = self.model.controlnet.controlnet_cond_embedding.conv_in.weight.grad
        self.assertGreater(float(grad[:,0].abs().sum()),0)
        self.assertEqual(float(grad[:,2].abs().sum()),0)
        self.assertGreater(float(self.model.fusion.sem_heads[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(self.model.source.layers[-1].weight.grad.abs().sum()),0)
        self.assertIsNone(self.model.unet.conv_in.weight.grad)
        self.assertTrue(torch.equal(frozen,self.model.unet.conv_in.weight))

    def test_reverse_cycle_gradient_and_rgb_replay_are_not_detached(self):
        x,p,q = conditions()
        reverse = nn.Sequential(nn.Conv2d(3,1,3,padding=1),nn.Tanh())
        rgb = torch.rand(1,3,64,64)*2-1
        tir = (reverse(rgb)+1)/2
        sx = torch.cat((tir,x[:,1:2],x[:,2:3]),1)
        generated = self.model(sx,p,q)
        F.l1_loss(generated,rgb).backward()
        self.assertGreater(float(reverse[0].weight.grad.abs().sum()),0)
        self.model.zero_grad(set_to_none=True)
        replay = self.model.replay_loss(rgb,x,p,q,torch.Generator("cpu").manual_seed(42))
        replay.backward()
        self.assertGreater(float(self.model.controlnet.controlnet_cond_embedding.conv_in.weight.grad.abs().sum()),0)

    def test_confidence_guard_and_signed_partial_checkpoint(self):
        x,p,q = conditions(); bad=x.clone();bad[:,2]=1
        with self.assertRaises(ValueError): self.model(bad,p,q)
        saved = self.model.export_state()
        self.assertNotIn("unet.conv_in.weight",saved)
        self.model.import_state(saved)
        altered = {k:v.clone() for k,v in saved.items()}
        altered["input_signature"][0] ^= 1
        with self.assertRaisesRegex(ValueError,"signature"):
            self.model.import_state(altered)

    def test_engine_skips_nonfinite_update_without_touching_weights(self):
        from D.unpaired.engine import UnpairedEngine
        x,p,q = conditions()
        class Builder:
            def synthetic(self, tir):
                return torch.cat(((tir+1)/2,x[:,1:2],x[:,2:3]),1),p,q
        class Teacher(nn.Module):
            def forward(self, rgb, side=256):
                return rgb.mean(1,keepdim=True).repeat(1,15,1,1).softmax(1)
        reverse = nn.Sequential(nn.Conv2d(3,1,3,padding=1),nn.Tanh())
        critics = nn.ModuleDict(dict(RGB=nn.Conv2d(3,1,1), TIR=nn.Conv2d(1,1,1)))
        config = copy.deepcopy(self.config); config["replay_every"] = 1
        engine = UnpairedEngine(self.model,reverse,critics,Builder(),Teacher(),config,device="cpu")
        before = self.model.export_state()
        backward_before = {k:v.clone() for k,v in reverse.state_dict().items()}
        sample = dict(x=x,p=p,q=q,rgb=torch.rand(1,3,64,64)*2-1)
        with patch("torch.nn.utils.clip_grad_norm_",return_value=torch.tensor(float("inf"))):
            applied, diagnostics = engine.update(sample,sample,0,10)
        self.assertFalse(applied)
        self.assertEqual(diagnostics["amp_overflow"],1)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in self.model.export_state().items()))
        self.assertTrue(all(torch.equal(backward_before[k],v) for k,v in reverse.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
