"""D-v3 quality: real gradient paths, degenerate colors, legacy and recovery contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from skimage.color import lab2rgb
import torch
from torch import nn

from D.common import ROOT, write_json
from D.unpaired import checkpoint
from D.unpaired.appearance import CLASSES, RegionAppearanceDistribution, lab_cdf, region_samples
from D.unpaired.engine import UnpairedEngine
from D.unpaired.model import FormalGenerator
from D.unpaired.protocol import config_from, IndexStream
from D.unpaired.quality import (APPEARANCE_LUMA_POLICY, APPEARANCE_REGION_POLICY, APPEARANCE_SYNTHETIC_TIR, augment_appearance_gray,
                                bounded_decode, bound_source_residual, color_attraction, rgb_appearance,
                                unsupported_structure, structure_quality)
from D.unpaired.train import training_config
from D.unpaired.tests.test_formal import bank, conditions
from D.unpaired.tests.test_review import fusion, fused
from D.tests import test_pipeline as tiny


def quality_config():
    return config_from(ROOT / "D/unpaired/config_quality_v3.json")


class QualityObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_saturated_pixels_keep_forward_values_and_receive_corrective_gradient(self):
        raw = torch.tensor([-1.4, -.4, .4, 1.4], requires_grad=True)
        output, penalty = bounded_decode(raw, straight_through=True)
        torch.testing.assert_close(output, raw.detach().clamp(-1, 1), rtol=0, atol=0)
        output.sum().backward(retain_graph=True)
        torch.testing.assert_close(raw.grad, torch.ones_like(raw))
        raw.grad = None; penalty.backward()
        self.assertLess(float(raw.grad[0]), 0); self.assertGreater(float(raw.grad[-1]), 0)
        self.assertEqual(float(raw.grad[1:3].abs().sum()), 0)

    def test_hard_policy_and_in_range_values_preserve_legacy_behavior(self):
        raw = torch.tensor([-2., -.5, .5, 2.], requires_grad=True)
        output, penalty = bounded_decode(raw)
        output.sum().backward()
        torch.testing.assert_close(raw.grad, torch.tensor([0., 1., 1., 0.]))
        self.assertGreater(float(penalty), 0)
        output, penalty = bounded_decode(torch.zeros(1), straight_through=True)
        self.assertEqual(float(penalty), 0)

    def test_straight_through_has_exact_clamped_forward_even_for_large_finite_values(self):
        raw=torch.tensor([-1e20,-1.1234567,0.,1.1234567,1e20],requires_grad=True)
        output,penalty=bounded_decode(raw,straight_through=True)
        torch.testing.assert_close(output,raw.detach().clamp(-1,1),rtol=0,atol=0)
        self.assertTrue(torch.isfinite(penalty))
        output.sum().backward();torch.testing.assert_close(raw.grad,torch.ones_like(raw))

    def test_source_budget_is_per_image_finite_and_does_not_suppress_small_residuals(self):
        raw = torch.cat((torch.full((1,4,8,8), .2), torch.full((1,4,8,8), 20.))).requires_grad_()
        bounded = bound_source_residual(raw, 1.)
        torch.testing.assert_close(bounded[0], raw[0])
        self.assertLessEqual(float(bounded[1].square().mean().sqrt()), 1.00001)
        bounded.square().mean().backward(); self.assertTrue(torch.isfinite(raw.grad).all())
        zero = torch.zeros(1,4,8,8, requires_grad=True)
        bound_source_residual(zero, 1.).sum().backward()
        torch.testing.assert_close(zero.grad, torch.ones_like(zero))
        self.assertTrue(torch.equal(bound_source_residual(raw, 0), raw))
        large=torch.full((1,4,8,8),60000.,dtype=torch.float16)
        small=bound_source_residual(large,.01)
        self.assertTrue(torch.isfinite(small).all());self.assertGreater(float(small.float().mean()),.009)

    def test_rgb_appearance_is_zero_for_the_correct_pair_and_has_source_color_gradients(self):
        settings = quality_config()["quality"]
        reference = torch.zeros(1,3,32,32); reference[:,0] = .4; reference[:,2] = -.2
        exact, _ = rgb_appearance(reference, reference, settings)
        self.assertEqual(float(exact), 0)
        gray = torch.zeros_like(reference, requires_grad=True)
        wrong, components = rgb_appearance(gray, reference.requires_grad_(), settings)
        self.assertGreater(float(wrong), 0); wrong.backward()
        self.assertGreater(float(gray.grad.abs().sum()), 0)
        self.assertIsNone(reference.grad)
        self.assertEqual(set(components), {"rgb_appearance_reconstruction", "rgb_appearance_chroma", "rgb_appearance_detail"})

    def test_region_appearance_weights_small_and_large_classes_equally(self):
        settings = quality_config()["quality"]
        person = torch.zeros(32, 32, dtype=torch.bool)
        person[4:8, 4:12] = True
        p = torch.zeros(1, 15, 32, 32)
        p[:, 8], p[:, 1] = person, ~person
        p.requires_grad_()
        q = torch.full((1, 1, 32, 32), .9, requires_grad=True)
        output = torch.zeros(1, 3, 32, 32, requires_grad=True)
        small = torch.zeros_like(output)
        large = torch.zeros_like(output)
        small[:, 0, person] = .6
        large[:, 0, ~person] = .6
        small.requires_grad_()

        original, _ = rgb_appearance(output, small, settings)
        balanced, components = rgb_appearance(output, small, settings, semantics=(p, q))
        _, large_components = rgb_appearance(output, large, settings, semantics=(p, q))
        self.assertEqual(float(components["rgb_appearance_region_classes"]), 2)
        for key in ("rgb_appearance_region_lightness", "rgb_appearance_region_chroma"):
            self.assertGreater(float(components[key]), 0)
            torch.testing.assert_close(components[key], large_components[key])
        self.assertGreater(float(balanced), float(original))
        balanced.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertIsNone(small.grad)
        self.assertIsNone(p.grad)
        self.assertIsNone(q.grad)

    def test_region_appearance_falls_back_for_background_unreliable_and_tiny_masks(self):
        settings = quality_config()["quality"]
        p = torch.zeros(1, 15, 32, 32)
        p[:, 0] = 1
        p[:, 0, :8, :8] = 0
        p[:, 7, :8, :8] = 1
        p[:, 0, -2:, -2:] = 0
        p[:, 9, -2:, -2:] = 1
        q = torch.ones(1, 1, 32, 32)
        q[:, :, :8, :8] = .49
        reference = torch.full((1, 3, 32, 32), .3)
        output = torch.zeros_like(reference, requires_grad=True)
        expected, expected_components = rgb_appearance(output, reference, settings)
        actual, components = rgb_appearance(output, reference, settings, semantics=(p, q))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in expected_components.items():
            torch.testing.assert_close(components[key], value, rtol=0, atol=0)
        self.assertEqual(float(components["rgb_appearance_region_classes"]), 0)
        actual.backward()
        self.assertTrue(torch.isfinite(output.grad).all())

    def test_appearance_luma_field_is_bounded_and_does_not_consume_global_rng(self):
        x, _, _ = conditions()
        x = x.repeat(2, 1, 1, 1)
        x[:, 0, :, 0] = 0
        x[:, 0, :, -1] = 1
        original = x.clone()
        rng = torch.get_rng_state()
        a = augment_appearance_gray(x, seed=42, step=3)
        torch.testing.assert_close(a, augment_appearance_gray(x, seed=42, step=3), rtol=0, atol=0)
        self.assertFalse(torch.equal(a, augment_appearance_gray(x, seed=42, step=7)))
        self.assertFalse(torch.equal(a[0, 0], a[1, 0]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(x, original))
        self.assertTrue(torch.equal(a[:, 1:], x[:, 1:]))
        self.assertTrue(torch.equal(a[:, 0, :, [0, -1]], x[:, 0, :, [0, -1]]))
        self.assertGreaterEqual(float(a.min()), 0)
        self.assertLessEqual(float(a.max()), 1)
        self.assertLessEqual(float((a[:, 0] - x[:, 0]).abs().max()), 0.150001)

    def test_continuous_mixture_color_loss_escapes_the_old_gray_dead_zone(self):
        from D.unpaired.semantics import legacy_models
        legacy_models()
        from models.learned_color_prior import LearnedChromaPrior
        from models.semantic_edges import FMB_CLASS_NAMES
        with tempfile.TemporaryDirectory() as temporary:
            path = bank(Path(temporary)/"prior.json")
            artifact = json.loads(path.read_text()); item = artifact["classes"]["road"]
            item.update(weights=[.7,.3], means_ab=[[18.,15.],[-18.,-15.]], variances_ab=[[16.,16.],[16.,16.]])
            item["loss_support"]["max_nll"] = 1000.
            path.write_text(json.dumps(artifact))
            prior = LearnedChromaPrior(path, FMB_CLASS_NAMES, require_loss_support=True)
            p = torch.zeros(1,15,16,16); p[:,1] = 1; q = torch.ones(1,1,16,16)
            gray = torch.zeros(1,3,16,16, requires_grad=True)
            old, _ = prior.chroma_loss(gray,p,q)
            self.assertEqual(float(old), 0)
            loss = color_attraction(prior,gray,p,q)
            self.assertGreater(float(loss), 0); loss.backward()
            self.assertGreater(float(gray.grad.abs().sum()), 0)
            for ab in ([18.,15.],[-18.,-15.]):
                rgb = lab2rgb(np.array([[[60.,*ab]]]))[0,0]
                mode = torch.tensor(rgb,dtype=torch.float32)[None,:,None,None].expand_as(gray)*2-1
                self.assertLess(float(color_attraction(prior,mode,p,q)), float(loss))
            zero = color_attraction(prior,gray,p,q*0)
            self.assertEqual(float(zero), 0)
            unsupported = p.clone(); unsupported[:,1]=0; unsupported[:,8]=1
            self.assertEqual(float(color_attraction(prior,gray,unsupported,q)), 0)

    def test_coarse_false_edges_are_penalized_but_unknown_and_supported_regions_are_exempt(self):
        tir = torch.full((1,1,64,64), .5); e = torch.zeros_like(tir)
        p = torch.zeros(1,15,64,64); p[:,1]=1; q=torch.ones_like(tir)
        generated = torch.zeros(1,3,64,64); generated[:,:,:,32:] = .9; generated.requires_grad_()
        loss = unsupported_structure(generated,tir,e,p,q)
        self.assertGreater(float(loss), 0); loss.backward()
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertGreater(float(generated.grad.abs().sum()), 0)
        self.assertEqual(float(unsupported_structure(generated,tir,e,p,q*0)),0)
        self.assertEqual(float(unsupported_structure(generated,tir,e+1,p,q)),0)
        self.assertEqual(float(unsupported_structure(torch.zeros_like(generated),tir,e,p,q)),0)

    def test_structural_context_keeps_color_quality_separate_and_unknown_data_is_isolated(self):
        q=torch.ones(1,1,8,8);q[:,:,3:5,3:5]=0;e=torch.zeros_like(q);e[:,:,3:5,3:5]=1
        restored = structure_quality(q,e)
        self.assertGreater(float(restored[:,:,3:5,3:5].sum()), 0)
        self.assertEqual(float(q[:,:,3:5,3:5].sum()), 0)
        module=fusion();module.boundary_context=True
        p=torch.zeros(1,15,8,8);p[:,0]=1
        changed=p.clone();changed[:,0,3:5,3:5]=0;changed[:,1,3:5,3:5]=1
        first=module.encode(p,q,(8,8),edge=e);second=module.encode(changed,q,(8,8),edge=e)
        for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertGreater(float(first[0][:,:,3:5,3:5].abs().sum()),0)
        self.assertEqual(float(first[3][:,:,3:5,3:5].sum()),0)
        empty=module.encode(changed,q*0,(8,8),edge=e)
        self.assertEqual(float(empty[2].sum()),0)


class QualityProtocolTests(unittest.TestCase):
    def check_config(self, value):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'config.json';write_json(path,value)
            return config_from(path)

    def test_quality_settings_are_explicit_and_cannot_enter_legacy_runs(self):
        value=quality_config();self.assertEqual(self.check_config(value)['format'],'D_unpaired_v3')
        for key,bad in [('appearance_every',0),('gradient_probe_every',False),('decoder_policy','guess'),
                        ('structure_min_quality',0),('source_residual_rms_limit',float('nan'))]:
            changed=copy.deepcopy(value);changed['quality'][key]=bad
            with self.assertRaises(ValueError):self.check_config(changed)
        changed=copy.deepcopy(value);changed['format']='D_unpaired_v1'
        with self.assertRaises(ValueError):self.check_config(changed)
        changed=copy.deepcopy(value);changed['color_min_quality']=0;changed['loss_weights']['color_prior']=0
        with self.assertRaises(ValueError):self.check_config(changed)

    def test_four_and_seven_step_presets_start_at_the_same_noise_level(self):
        from diffusers import DDIMScheduler
        a=quality_config();b=config_from(ROOT/'D/unpaired/config_quality_v3_7step.json')
        scheduler=DDIMScheduler(num_train_timesteps=1000,steps_offset=1,set_alpha_to_one=False)
        starts=[];counts=[]
        for c in (a,b):
            scheduler.set_timesteps(c['sampling_steps'])
            t=scheduler.timesteps[-int(c['sampling_steps']*c['strength']):]
            starts.append(int(t[0]));counts.append(len(t))
        self.assertEqual(starts,[301,301]);self.assertEqual(counts,[4,7])

    def test_luma_augmentation_is_recorded_only_for_active_v3_appearance_training(self):
        path = ROOT / "D/unpaired/config_quality_v3.json"
        clean = training_config(path)
        aug = training_config(path, augment_luma=True)
        self.assertEqual(aug.pop("rgb_appearance_luma_augment"), APPEARANCE_LUMA_POLICY)
        self.assertEqual(aug, clean)
        with self.assertRaisesRegex(ValueError, "requires the D-v3 RGB appearance loss"):
            training_config(ROOT / "D/unpaired/config.json", augment_luma=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "no_appearance.json"
            clean["loss_weights"]["rgb_appearance"] = 0
            write_json(path, clean)
            with self.assertRaisesRegex(ValueError, "requires the D-v3 RGB appearance loss"):
                training_config(path, augment_luma=True)

    def test_region_balance_is_an_opt_in_training_policy(self):
        path = ROOT / "D/unpaired/config_quality_v3.json"
        clean = training_config(path)
        enabled = training_config(path, balance_regions=True)
        self.assertEqual(enabled.pop("rgb_appearance_region_balance"), APPEARANCE_REGION_POLICY)
        self.assertEqual(enabled, clean)
        with self.assertRaisesRegex(ValueError, "RGB appearance region balance"):
            training_config(ROOT / "D/unpaired/config.json", balance_regions=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "detail_only.json"
            clean["quality"].update(appearance_rgb=0, appearance_chroma=0)
            write_json(path, clean)
            with self.assertRaisesRegex(ValueError, "RGB appearance region balance"):
                training_config(path, balance_regions=True)


class QualityGeneratorTests(unittest.TestCase):
    def setUp(self):
        tiny.TinyPipelineTests.setUpClass()
        self.temp=tempfile.TemporaryDirectory()
        self.path=bank(Path(self.temp.name)/'prior.json')
        self.config=quality_config()
        self.config.update(width=64,height=64,sampling_steps=4,strength=.75,lora_rank=2)
        self.config['quality'].update(appearance_every=1,gradient_probe_every=1)
        self.model=FormalGenerator(tiny.TinyPipelineTests.pipeline,self.config,self.path,{'fixture':'quality'})

    def tearDown(self):
        self.temp.cleanup();tiny.TinyPipelineTests.tearDownClass()

    def engine(self, *, appearance_only=False, distribution=False):
        x,p,q=conditions()
        class Builder:
            def synthetic(self,tir):
                return torch.cat(((tir+1)/2,x[:,1:2],x[:,2:3]),1),p,q
        class Teacher(nn.Module):
            def forward(self,rgb,side=256):
                return rgb.mean(1,keepdim=True).repeat(1,15,1,1).softmax(1)
        reverse=nn.Sequential(nn.Conv2d(3,1,3,padding=1),nn.Tanh())
        critics=nn.ModuleDict(dict(RGB=nn.Conv2d(3,1,1),TIR=nn.Conv2d(1,1,1)))
        config=copy.deepcopy(self.config)
        if appearance_only:
            config['loss_weights']={name:0. for name in config['loss_weights']}
            config['loss_weights'].update(rgb_appearance=2.,decoder_range=.1)
        sample=dict(x=x,p=p,q=q,rgb=torch.rand(1,3,64,64)*1.6-.8)
        appearance = None
        if distribution:
            config.update(appearance_distribution_weight=.1, appearance_chroma_detail=.5,
                          rgb_appearance_input=APPEARANCE_SYNTHETIC_TIR,
                          rgb_appearance_region_balance=APPEARANCE_REGION_POLICY)
            config["loss_weights"].update(color_prior=0., color_attraction=0.)
            lab, active = region_samples(sample["rgb"], p, q, config["color_min_quality"])
            reference = (lab_cdf(lab)[0] * active[0, :, None, None], active[0], "fixture")
            appearance = RegionAppearanceDistribution(reference, config["color_min_quality"], "cpu")
        engine=UnpairedEngine(self.model,reverse,critics,Builder(),Teacher(),config,device='cpu',
                              appearance_distribution=appearance)
        return engine,sample

    def test_actual_rgb_forward_reaches_source_and_excludes_confidence(self):
        x,p,q=conditions()
        output,aux=self.model(x,p,q,return_aux=True)
        target=torch.ones_like(output)*.25
        loss,_=rgb_appearance(output,target,self.config['quality'])
        (loss+.1*aux['decoder_range']).backward()
        self.assertGreater(float(self.model.source.layers[-1].weight.grad.abs().sum()),0)
        self.assertEqual(float(self.model.source.layers[0].weight.grad[:,2].abs().sum()),0)
        self.assertIsNone(self.model.unet.conv_in.weight.grad)
        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(float(output.max()),1);self.assertGreaterEqual(float(output.min()),-1)

    def test_legacy_and_different_sampling_signatures_are_rejected(self):
        state=self.model.export_state();changed=dict(self.config,sampling_steps=5)
        # Build a fresh pipeline, so trainable adapters never get installed twice.
        tiny.TinyPipelineTests.tearDownClass();tiny.TinyPipelineTests.setUpClass()
        other=FormalGenerator(tiny.TinyPipelineTests.pipeline,changed,self.path,{'fixture':'quality'})
        with self.assertRaisesRegex(ValueError,'signature'):other.import_state(state)
        legacy=config_from();legacy.update(width=64,height=64,sampling_steps=4,strength=.75,lora_rank=2)
        tiny.TinyPipelineTests.tearDownClass();tiny.TinyPipelineTests.setUpClass()
        old=FormalGenerator(tiny.TinyPipelineTests.pipeline,legacy,self.path,{'fixture':'quality'})
        with self.assertRaisesRegex(ValueError,'signature'):old.import_state(state)

    def test_complete_quality_update_is_finite_and_weighted_total_matches(self):
        engine,sample=self.engine()
        applied,values=engine.update(sample,sample,3,8,diagnostics=True)
        self.assertTrue(applied)
        for name in ('rgb_appearance','color_attraction','edge_precision','decoder_range',
                     'decoder_range_cycle','decoder_range_rgb','rgb_appearance_chroma',
                     'objective_probe_grad_rgb_appearance','clip_norm_diffusion','clip_norm_reverse','rgb_replay'):
            self.assertIn(name,values);self.assertTrue(torch.isfinite(values[name]).all(),name)
        total=sum(value for key,value in values.items() if key.startswith('weighted_'))
        torch.testing.assert_close(values['G_total'],total)
        self.assertGreater(float(values['objective_probe_grad_rgb_appearance']),0)
        self.assertEqual(float(values['grad_C_zero']),0)

    def test_appearance_alone_updates_source_adapter(self):
        engine,sample=self.engine(appearance_only=True)
        before=self.model.source.layers[-1].weight.detach().clone()
        applied,values=engine.update(sample,sample,0,4,diagnostics=True)
        self.assertTrue(applied)
        self.assertGreater(float(values['grad_content_adapter']),0)
        self.assertFalse(torch.equal(before,self.model.source.layers[-1].weight))
        self.assertEqual(values['rgb_appearance_active'],1)

    def test_failure_in_appearance_graph_retries_without_partial_commit_or_noise_advance(self):
        engine,sample=self.engine();before=engine.retry_state()
        engine.config["rgb_appearance_luma_augment"] = APPEARANCE_LUMA_POLICY
        state={k:v.clone() for k,v in engine.generator.export_state().items()}
        original=engine._require_finite
        def fail(values,stage):
            if stage=='rgb_appearance_forward':raise FloatingPointError('forced appearance failure')
            return original(values,stage)
        with patch.object(engine,'_require_finite',side_effect=fail):
            applied,values=engine.update(sample,sample,0,4,diagnostics=True)
        self.assertFalse(applied);self.assertFalse(engine.opt_g.state);self.assertFalse(engine.opt_d.state)
        self.assertTrue(torch.equal(before['noise_rng'],engine.noise_rng.get_state()))
        for name,value in engine.generator.export_state().items():self.assertTrue(torch.equal(value,state[name]))
        applied,values=engine.update(sample,sample,0,4,diagnostics=True)
        self.assertTrue(applied)
        self.assertGreater(float(values["rgb_appearance_luma_delta"]), 0)

    def test_full_state_resume_repeats_the_next_quality_update_exactly(self):
        engine,sample=self.engine();streams=dict(TIR=IndexStream(3,'TIR'),RGB=IndexStream(4,'RGB'))
        engine.config["rgb_appearance_luma_augment"] = APPEARANCE_LUMA_POLICY
        engine.config["rgb_appearance_region_balance"] = APPEARANCE_REGION_POLICY
        for step in range(3):
            applied,_=engine.update(sample,sample,step,8,diagnostics=True);self.assertTrue(applied)
        folder=Path(self.temp.name)/'run';folder.mkdir();write_json(folder/'run.json',{'fixture':'quality'})
        location=checkpoint.save(folder,3,engine,streams)
        applied,_=engine.update(sample,sample,3,8,diagnostics=True);self.assertTrue(applied)
        expected=[{k:v.detach().clone() for k,v in module.state_dict().items()}
                  for module in (engine.generator,engine.reverse,engine.critics)]
        self.assertEqual(checkpoint.restore(folder,location,engine,streams),3)
        applied,_=engine.update(sample,sample,3,8,diagnostics=True);self.assertTrue(applied)
        for module,wanted in zip((engine.generator,engine.reverse,engine.critics),expected):
            for name,value in module.state_dict().items():torch.testing.assert_close(value,wanted[name],rtol=0,atol=0)

    def test_gradient_probes_do_not_change_the_optimization_or_noise_stream(self):
        engine,sample=self.engine();other=copy.deepcopy(engine)
        before=checkpoint.capture_rng()
        applied,_=engine.update(sample,sample,3,8,diagnostics=False);self.assertTrue(applied)
        checkpoint.restore_rng(before)
        applied,_=other.update(sample,sample,3,8,diagnostics=True);self.assertTrue(applied)
        for left,right in zip((engine.generator,engine.reverse,engine.critics),(other.generator,other.reverse,other.critics)):
            for name,value in left.state_dict().items():torch.testing.assert_close(value,right.state_dict()[name],rtol=0,atol=0)
        self.assertTrue(torch.equal(engine.noise_rng.get_state(),other.noise_rng.get_state()))
        self.assertTrue(torch.equal(engine.patch_rng.get_state(),other.patch_rng.get_state()))

    def test_appearance_uses_the_rgb_samples_own_gray_condition(self):
        engine,sample=self.engine(appearance_only=True)
        tir=dict(sample,x=sample['x'].clone());tir['x'][:,:1]*=.5
        with patch.object(self.model,'forward',wraps=self.model.forward) as forward:
            applied,_=engine.update(tir,sample,0,4,diagnostics=True)
        self.assertTrue(applied);self.assertEqual(forward.call_count,2)
        self.assertTrue(torch.equal(forward.call_args_list[-1].args[0],sample['x']))
        self.assertFalse(torch.equal(forward.call_args_list[-1].args[0],tir['x']))

    def test_region_appearance_uses_the_rgb_samples_own_semantic_cache(self):
        engine, rgb = self.engine(appearance_only=True)
        engine.config["rgb_appearance_region_balance"] = APPEARANCE_REGION_POLICY
        tir = dict(rgb, q=torch.zeros_like(rgb["q"]))
        with (
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
            patch("D.unpaired.engine.rgb_appearance", wraps=rgb_appearance) as loss,
        ):
            applied, values = engine.update(tir, rgb, 0, 4, diagnostics=True)
        self.assertTrue(applied)
        self.assertEqual(forward.call_count, 2)
        self.assertIs(loss.call_args.kwargs["semantics"][0], rgb["p"])
        self.assertIs(loss.call_args.kwargs["semantics"][1], rgb["q"])
        self.assertEqual(float(values["rgb_appearance_region_classes"]), 1)
        self.assertGreater(float(values["rgb_appearance_region_chroma"]), 0)
        self.assertGreater(float(values["grad_content_adapter"]), 0)

    def test_pseudo_appearance_reuses_synthetic_inputs_without_reverse_gradients(self):
        engine, rgb = self.engine(distribution=True)
        cycle_builder = engine.builder.synthetic

        def distinguish_conditions(fake_tir):
            x, p, q = cycle_builder(fake_tir)
            return x, p.roll(1, dims=1), q * .75

        def check_loss(generated, target, settings, **kwargs):
            loss, components = rgb_appearance(generated, target, settings, **kwargs)
            reverse_grads = torch.autograd.grad(loss, tuple(engine.reverse.parameters()),
                                                allow_unused=True, retain_graph=True)
            self.assertTrue(all(grad is None for grad in reverse_grads))
            self.assertIs(target, rgb["rgb"])
            self.assertIs(kwargs["semantics"][0], rgb["p"])
            self.assertIs(kwargs["semantics"][1], rgb["q"])
            return loss, components

        with (
            patch.object(engine.builder, "synthetic", side_effect=distinguish_conditions) as build,
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
            patch("D.unpaired.engine.rgb_appearance", side_effect=check_loss),
        ):
            applied, values = engine.update(rgb, rgb, 0, 4, diagnostics=True)
        self.assertTrue(applied)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(forward.call_count, 3)
        cycle_inputs, appearance_inputs = (call.args for call in forward.call_args_list[-2:])
        self.assertTrue(cycle_inputs[0].requires_grad)
        for original, reused in zip(cycle_inputs, appearance_inputs):
            self.assertFalse(reused.requires_grad)
            torch.testing.assert_close(original, reused, rtol=0, atol=0)
        self.assertFalse(torch.equal(rgb["p"], appearance_inputs[1]))
        self.assertEqual(values["rgb_appearance_synthetic"], 1)
        self.assertGreater(float(values["rgb_appearance_chroma_detail"]), 0)
        self.assertGreater(float(values["appearance_dist_classes"]), 0)
        self.assertNotIn("color_prior", values)
        self.assertNotIn("color_attraction", values)
        torch.testing.assert_close(values["G_total"], sum(v for k, v in values.items() if k.startswith("weighted_")))

    def test_distribution_history_commits_once_and_is_restored(self):
        engine, sample = self.engine(distribution=True)
        distribution = engine.appearance_distribution
        initial = distribution.history.clone()
        self.assertEqual(initial.shape[0], len(CLASSES))
        original = engine._require_finite

        def fail(values, stage):
            if stage == "rgb_appearance_forward":
                raise FloatingPointError("forced precommit failure")
            return original(values, stage)

        with patch.object(engine, "_require_finite", side_effect=fail):
            self.assertFalse(engine.update(sample, sample, 0, 4)[0])
        torch.testing.assert_close(distribution.history, initial, rtol=0, atol=0)
        self.assertTrue(engine.update(sample, sample, 0, 4)[0])
        committed = distribution.history.clone()
        self.assertFalse(torch.equal(initial, committed))
        folder = Path(self.temp.name) / "appearance_run"
        folder.mkdir()
        write_json(folder / "run.json", {"fixture": "appearance"})
        streams = {"TIR": IndexStream(2, "TIR"), "RGB": IndexStream(2, "RGB")}
        location = checkpoint.save(folder, 1, engine, streams)
        self.assertTrue(engine.update(sample, sample, 1, 4)[0])
        next_history = distribution.history.clone()
        expected = {k: v.clone() for k, v in engine.generator.export_state().items()}
        self.assertEqual(checkpoint.restore(folder, location, engine, streams), 1)
        torch.testing.assert_close(distribution.history, committed, rtol=0, atol=0)
        self.assertTrue(engine.update(sample, sample, 1, 4)[0])
        torch.testing.assert_close(distribution.history, next_history, rtol=0, atol=0)
        for name, value in engine.generator.export_state().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_luma_augmentation_isolated_to_rgb_appearance_input_and_noise_is_matched(self):
        engine, rgb = self.engine()
        control = copy.deepcopy(engine)
        engine.config["rgb_appearance_luma_augment"] = APPEARANCE_LUMA_POLICY
        tir = dict(rgb, x=rgb["x"].clone())
        tir["x"][:, :1] *= 0.5
        original = rgb["x"].clone()
        step = engine.config["replay_every"] - 1
        global_rng = checkpoint.capture_rng()
        applied, clean = control.update(tir, rgb, step, 8, diagnostics=True)
        self.assertTrue(applied)
        self.assertNotIn("rgb_appearance_luma_delta", clean)
        checkpoint.restore_rng(global_rng)
        with (
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
            patch.object(self.model, "replay_loss", wraps=self.model.replay_loss) as replay,
            patch("D.unpaired.engine.rgb_appearance", wraps=rgb_appearance) as loss,
        ):
            applied, aug = engine.update(tir, rgb, step, 8, diagnostics=True)
        self.assertTrue(applied)
        self.assertGreater(float(aug["rgb_appearance_luma_delta"]), 0)
        torch.testing.assert_close(forward.call_args_list[0].args[0], tir["x"], rtol=0, atol=0)
        appearance_x, appearance_p, appearance_q = forward.call_args_list[-1].args
        expected = augment_appearance_gray(original, seed=engine.config["seed"], step=step)
        torch.testing.assert_close(appearance_x, expected, rtol=0, atol=0)
        self.assertIs(appearance_p, rgb["p"])
        self.assertIs(appearance_q, rgb["q"])
        self.assertIs(loss.call_args.args[1], rgb["rgb"])
        self.assertIs(replay.call_args.args[1], rgb["x"])
        self.assertTrue(torch.equal(rgb["x"], original))
        self.assertTrue(torch.equal(engine.noise_rng.get_state(), control.noise_rng.get_state()))
        self.assertTrue(torch.equal(engine.patch_rng.get_state(), control.patch_rng.get_state()))

    def test_quality_switches_can_reproduce_the_legacy_forward_exactly(self):
        x,p,q=conditions();q[:,:,::4]=0
        with torch.no_grad():
            self.model.source.layers[-1].weight.normal_(0,.005)
            for layer in [*self.model.fusion.sem_heads,*self.model.fusion.color_heads]:layer.weight.normal_(0,.005)
        legacy=config_from();legacy.update(width=64,height=64,sampling_steps=4,strength=.75,lora_rank=2)
        self.model.config=legacy;self.model.quality_v3=False
        self.model.fusion.policy='legacy';self.model.fusion.boundary_context=False
        expected=self.model(x,p,q).detach()
        control=copy.deepcopy(self.config);control['quality'].update(decoder_policy='hard',boundary_context=False,source_residual_rms_limit=0.)
        control['fusion_policy']='legacy';control['chroma_smooth_policy']='legacy'
        self.model.config=control;self.model.quality_v3=True
        actual=self.model(x,p,q).detach()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
