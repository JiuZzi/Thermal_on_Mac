"""Learned-token gradients, exact warm start, Q limits and checkpoint identity."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models import create_model,networks
from models.learned_color import LearnedColorGenerator
from models.learned_color_prior import LearnedChromaPrior,semantic_reliability
from models.semantic_color import SemanticColorGenerator
from test_semantic_contour_training import FakePrior,fixture,options as base_options


def bank_file(root,shift=0):
    bank=dict(format='learned_chroma_gmm_v1',class_names=['vegetation','sky'],classes={
        'vegetation':dict(status='supported',weights=[.5,.5],means_ab=[[-20+shift,25],[15+shift,25]],variances_ab=[[4,9],[9,4]]),
        'sky':dict(status='unsupported')})
    path=root/f'bank_{shift}.json'; path.write_text(json.dumps(bank)); return path


def options(root,bank,**changes):
    values=dict(color_condition='semantic',color_width=4,lambda_color_prior=0.,init_semantic_dir=None,init_semantic_epoch='80',
                learned_color_prior=str(bank),learned_color_mode='learned',learned_color_width=4,
                learned_color_dropout=.1,learned_color_weight_decay=.0001,reliability_flip=True,
                learned_color_loss_weight=0.,learned_color_loss_min_quality=.5,learned_color_loss_warmup_steps=1000,
                reliability_boundary_radius=1,gate_semantic_conditions=False,init_condition_dir=None,
                init_condition_epoch='5',color_feature_override='normal')
    values.update(changes)
    return base_options(root,model='learned_color_cycle_gan',**values)


class LearnedColorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        stub=patch('models.semantic_contour.FrozenSemanticPrior',FakePrior)
        stub.start(); self.addCleanup(stub.stop)

    def generator(self,path,**kwargs):
        torch.manual_seed(42)
        backbone=networks.ResnetGenerator(3,3,ngf=8,n_blocks=1,norm_layer=networks.get_norm_layer('instance'))
        semantic=SemanticColorGenerator(backbone,'stub.pth',color_condition='semantic',color_width=4)
        return LearnedColorGenerator(semantic,path,width=4,dropout=0.,**kwargs)

    def test_quality_handles_ties_view_disagreement_and_class_boundaries(self):
        p=torch.zeros(1,3,9,13); p[:,0]=1
        q=semantic_reliability(p,boundary_radius=1)
        self.assertTrue(torch.equal(q,torch.ones_like(q)))
        # A consistent high-confidence mistake still passes Q: a documented limit.
        self.assertEqual(semantic_reliability(p,p,boundary_radius=0).min(),1)
        opposite=p.roll(1,1)
        self.assertEqual(semantic_reliability(p,opposite,boundary_radius=0).count_nonzero(),0)
        self.assertEqual(semantic_reliability((p+opposite)/2,boundary_radius=0).count_nonzero(),0)
        p[:,:,:,6:]=opposite[:,:,:,6:]
        q=semantic_reliability(p,boundary_radius=1)
        self.assertEqual(q[:,:,:,4:8].count_nonzero(),0)
        with self.assertRaises(ValueError): semantic_reliability(p*.5)

    def test_modes_are_not_averaged_and_mapping_is_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=bank_file(root)
            prior=LearnedChromaPrior(path,['sky','vegetation','other'])
            p=torch.zeros(1,3,7,11); p[:,1]=1
            result=prior(p,semantic_reliability(p,boundary_radius=0))
            self.assertEqual(result['features'].shape,(1,10,7,11))
            self.assertGreater(result['features'][:,1].abs().sum(),0)
            self.assertGreater(result['features'][:,6].abs().sum(),0)
            p=p.roll(-1,1)
            self.assertEqual(prior(p,torch.ones_like(p[:,:1]))['strength'].count_nonzero(),0)

    def test_new_zero_initialized_branch_preserves_trained_semantic_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator=self.generator(bank_file(Path(tmp)))
            # Nonzero semantic projections emulate an already-trained semantic model.
            for layer in generator.semantic.condition_projections.values():
                torch.nn.init.normal_(layer.weight,0,.01)
            generator.eval(); generator.semantic.eval()
            with torch.no_grad():
                expected=generator.semantic(fixture())
                actual,aux=generator(fixture(),return_aux=True)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertFalse(aux['quality'].requires_grad)

    def test_branch_learns_and_off_override_removes_bias_contribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator=self.generator(bank_file(Path(tmp)))
            optimizer=torch.optim.Adam(generator.parameters(),lr=.005)
            for _ in range(3):
                optimizer.zero_grad(); rgb=generator(fixture()); rgb.square().mean().backward(); optimizer.step()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in generator.learned_encoder.parameters()),0)
            self.assertTrue(all(p.grad is None for p in generator.semantic_prior.model.parameters()))
            generator.eval()
            with torch.no_grad():
                normal=generator(fixture()); generator.feature_override='off'; off=generator(fixture())
                expected=generator.semantic(fixture())
            torch.testing.assert_close(off,expected,atol=1e-6,rtol=1e-6)
            self.assertGreater((normal-off).abs().max(),1e-5)

    def test_q_zero_disables_projection_even_after_it_learns_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator=self.generator(bank_file(Path(tmp)))
            for layer in generator.learned_projections.values(): layer.bias.data.fill_(2.)
            p=torch.full((1,15,32,40),1/15)
            generator.eval()
            with torch.no_grad():
                rgb=generator(fixture(),probabilities_override=p)
                generator.feature_override='off'
                off=generator(fixture(),probabilities_override=p)
            torch.testing.assert_close(rgb,off,atol=0,rtol=0)

    def test_changed_bank_or_control_mode_cannot_load_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=bank_file(root)
            state=self.generator(path).state_dict()
            for other in [self.generator(bank_file(root,1)),self.generator(path,mode='zero_tokens')]:
                with self.assertRaisesRegex(RuntimeError,'signature mismatch'): other.load_state_dict(state)

    def test_optional_semantic_gate_also_blocks_unreliable_old_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator=self.generator(bank_file(Path(tmp)),gate_semantic=True)
            for layer in generator.semantic.condition_projections.values(): layer.bias.data.fill_(2.)
            generator.eval()
            p=torch.full((1,15,32,40),1/15)
            with torch.no_grad():
                rgb,aux=generator(fixture(),probabilities_override=p,return_aux=True)
                expected=generator.backbone(torch.cat((fixture(),aux['edge'],torch.zeros_like(fixture())),1))
            torch.testing.assert_close(rgb,expected,atol=0,rtol=0)

    def test_full_cycle_update_roundtrip_and_semantic_checkpoint_warmstart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bank=bank_file(root)
            model=create_model(options(root,bank)); model.setup(model.opt)
            tir=fixture()[:,:,:,:32]
            data=dict(B=tir,A=tir.repeat(1,3,1,1),B_paths=['tir'],A_paths=['rgb'])
            model.set_input(data); model.optimize_parameters()
            self.assertTrue(all(np.isfinite(x) for x in model.get_current_losses().values()))
            self.assertEqual(model.loss_color_prior.item(),0)
            self.assertEqual(len(model.netG_A.semantic_prior.inputs),4)
            model.eval(); model.test(); expected=model.fake_B.clone()
            model.save_networks('latest')
            loaded=create_model(options(root,bank,isTrain=False)); loaded.setup(loaded.opt)
            loaded.eval(); loaded.set_input(data); loaded.test()
            torch.testing.assert_close(expected,loaded.fake_B,atol=1e-6,rtol=1e-6)
            old=root/'semantic_only'; old.mkdir()
            torch.save(model.netG_A.semantic.state_dict(),old/'5_net_G_A.pth')
            torch.save(model.netG_B.state_dict(),old/'5_net_G_B.pth')
            warm=create_model(options(root,bank,name='warm',init_condition_dir=str(old)))
            warm.setup(warm.opt)
            torch.testing.assert_close(warm.netG_A.semantic.state_dict(),model.netG_A.semantic.state_dict(),atol=0,rtol=0)

    def test_tiny_color_loss_updates_and_saved_warmup_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bank=bank_file(root)
            content=json.loads(bank.read_text())
            content['classes']['vegetation']['loss_support']=dict(source='group_balanced_train_only',quantile=.99,max_nll=10.)
            bank.write_text(json.dumps(content))
            opt=options(root,bank,learned_color_loss_weight=.01,learned_color_loss_warmup_steps=2)
            model=create_model(opt); model.setup(opt)
            tir=fixture()[:,:,:,:32]
            data=dict(B=tir,A=tir.repeat(1,3,1,1),B_paths=['tir'],A_paths=['rgb'])
            model.set_input(data); model.optimize_parameters()
            self.assertGreater(float(model.loss_color_prior_raw),0)
            self.assertGreater(float(model.loss_color_coverage),0)
            self.assertAlmostEqual(model.loss_color_weight,.005)
            self.assertGreater(float(model.loss_color_prior),0)
            self.assertLessEqual(float(model.loss_color_prior),.005)
            self.assertTrue(all(np.isfinite(x) for x in model.get_current_losses().values()))
            self.assertTrue(all(p.grad is None for p in model.netG_A.semantic_prior.model.parameters()))
            model.save_networks('latest')
            loaded=create_model(options(root,bank,isTrain=False,learned_color_loss_weight=.01,learned_color_loss_warmup_steps=2))
            loaded.setup(loaded.opt)
            self.assertEqual(int(loaded.netG_A.color_loss_updates),1)
            model.set_input(data); model.optimize_parameters()
            self.assertAlmostEqual(model.loss_color_weight,.01)
            with self.assertRaisesRegex(RuntimeError,'signature mismatch'):
                loaded.netG_A.load_state_dict(self.generator(bank).state_dict())

    def test_fresh_gan_setup_does_not_load_previous_generator_or_discriminator_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bank=bank_file(root)
            model=create_model(options(root,bank))
            # FakePrior supplies frozen segmentation weights in this test. Any
            # torch.load in GAN setup would therefore be an accidental warm start.
            with patch('torch.load',side_effect=AssertionError('Unexpected GAN checkpoint load')):
                model.setup(model.opt)
            metadata=json.loads((model.save_dir/'color_train.json').read_text())
            self.assertEqual(metadata['initialization'],'fresh')
            self.assertNotIn('sources',metadata)


if __name__=='__main__': unittest.main()
