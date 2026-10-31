import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from D.common import write_json, sha256
from D.diagnostics.edge_use import ProbeSettings, interventions, probe_maps, VARIANTS
from D.diagnostics.compare_edge_use import ProbeDataset, compare_samples, rgb_metrics, main


def sample(size=48):
    y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    t=(.2+.35*(x>size//2)+.04*torch.sin(y.float()*2.2))[None,None]
    E=((x%4==0)|(y%5==0)).float()[None,None]*.7
    p=torch.zeros(1,15,size,size);p[:,0,:,:size//2]=1;p[:,1,:,size//2:]=1
    q=torch.ones_like(t)
    return torch.cat((t,E,torch.zeros_like(E)),1),p,q


def cache(folder):
    x,p,q=sample();folder.mkdir();(folder/'maps').mkdir()
    relative='maps/fixture.npz'
    np.savez_compressed(folder/relative,gray=(x[0,0].numpy()*255).round().astype(np.uint8),E=x[0,1].numpy(),P=p[0].numpy(),Q=q[0,0].numpy())
    frame=dict(id='fixture',stem='fixture',filename='fixture.png',condition=relative,condition_sha256=sha256(folder/relative),native_size=[48,48],source='fixture.png',source_sha256='fixture')
    write_json(folder/'manifest.json',dict(format='edge_use_probe_inputs_v1',phase='train',frames=[frame]))
    write_json(folder/'status.json',dict(phase='completed',completed=1,total=1))
    return ProbeDataset(folder)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_uniform_and_selective_have_same_actual_resized_edge_mass(self):
        x,p,q=sample()
        variants,maps,stats=interventions(x,p,q,model_size=(64,80))
        self.assertAlmostEqual(stats[0]['selective_model_mass'],stats[0]['uniform_model_mass'],places=4)
        self.assertFalse(torch.equal(variants['uniform'][:,1],variants['selective'][:,1]))
        self.assertTrue(.25<=stats[0]['uniform_gain']<=1.)

    def test_original_TIR_E_P_Q_and_reserved_channel_are_not_mutated(self):
        x,p,q=sample();before=[a.clone() for a in (x,p,q)]
        variants,_,_=interventions(x,p,q)
        for a,b in zip((x,p,q),before):self.assertTrue(torch.equal(a,b))
        for value in variants.values():
            self.assertTrue(torch.equal(value[:,:1],x[:,:1]));self.assertEqual(int(torch.count_nonzero(value[:,2])),0)

    def test_every_probe_is_attenuation_only(self):
        x,p,q=sample();variants,_,_=interventions(x,p,q)
        for value in variants.values():self.assertTrue((value[:,1]<=x[:,1]+1e-7).all())

    def test_confident_semantic_boundary_is_protected_and_interior_is_reduced(self):
        x,p,q=sample();maps=probe_maps(x,p,q)
        boundary=maps['semantic_boundary']>.99
        self.assertTrue(boundary.any());self.assertTrue(torch.allclose(maps['gate'][boundary],torch.ones_like(maps['gate'][boundary])))
        self.assertLess(float(maps['gate'][:,:,10:35,5:15].mean()),.95)

    def test_zero_semantic_quality_abstains_from_attenuation(self):
        x,p,q=sample();maps=probe_maps(x,p,q*0)
        self.assertTrue(torch.equal(maps['gate'],torch.ones_like(q)))

    def test_no_semantic_class_is_hard_deleted(self):
        x,p,q=sample()
        for index in (0,1,6,7,8,9):
            p.zero_();p[:,index]=1
            maps=probe_maps(x,p,q)
            self.assertGreaterEqual(float(maps['gate'].min()),.25)

    def test_class_permutation_does_not_change_the_rule(self):
        x,p,q=sample();a=probe_maps(x,p,q);b=probe_maps(x,p.flip(1),q)
        torch.testing.assert_close(a['gate'],b['gate'],rtol=0,atol=1e-6)

    def test_polarity_and_ninety_degree_rotation_do_not_encode_scene_position(self):
        x,p,q=sample();a=probe_maps(x,p,q)['gate']
        inverted=x.clone();inverted[:,:1]=1-inverted[:,:1]
        b=probe_maps(inverted,p,q)['gate'];torch.testing.assert_close(a,b,atol=2e-6,rtol=0)
        b=probe_maps(x.rot90(1,(-2,-1)),p.rot90(1,(-2,-1)),q.rot90(1,(-2,-1)))['gate']
        torch.testing.assert_close(a,b.rot90(-1,(-2,-1)),atol=2e-6,rtol=0)

    def test_each_batch_item_uses_its_own_noise_and_strength_statistics(self):
        x,p,q=sample();second=x.clone();second[:,:1]=.5
        together=probe_maps(torch.cat((x,second)),p.repeat(2,1,1,1),q.repeat(2,1,1,1))['gate']
        torch.testing.assert_close(together[:1],probe_maps(x,p,q)['gate'],atol=1e-6,rtol=0)
        torch.testing.assert_close(together[1:],probe_maps(second,p,q)['gate'],atol=1e-6,rtol=0)

    def test_empty_E_has_finite_identity_control(self):
        x,p,q=sample();x[:,1]=0
        variants,maps,stats=interventions(x,p,q)
        self.assertEqual(stats[0]['uniform_gain'],1.)
        self.assertTrue(all(torch.equal(value,x) for value in variants.values()))
        json.dumps(stats,allow_nan=False)

    def test_invalid_probabilities_or_nonzero_third_channel_are_rejected(self):
        x,p,q=sample()
        with self.assertRaises(ValueError):probe_maps(x,p*.8,q)
        bad=x.clone();bad[:,2]=.2
        with self.assertRaises(ValueError):probe_maps(bad,p,q)
        bad=x.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):probe_maps(bad,p,q)

    def test_rules_are_frozen_and_not_learned_confidence(self):
        x,p,q=sample();x.requires_grad_()
        _,maps,stats=interventions(x,p,q)
        self.assertFalse(maps['gate'].requires_grad)
        self.assertFalse(stats[0]['gate_is_learned']);self.assertFalse(stats[0]['correctness_probability'])
        with self.assertRaises(ValueError):ProbeSettings(suppression=1.)
        with self.assertRaises(ValueError):ProbeSettings(window_fraction=float('nan'))


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.dataset=cache(self.root/'inputs');self.output=self.root/'output'
        self.config=dict(height=48,width=48);self.identity=dict(kind='pretrained_only',fixture=True)
        self.calls=[]

    def tearDown(self):self.tmp.cleanup()

    def predictor(self,x,p,q):
        self.calls.append((x.clone(),p.clone(),q.clone()))
        # Deterministic synthetic renderer for transaction tests, not real RGB.
        t=x[0,0].numpy();e=x[0,1].numpy()
        rgb=np.stack((t*.8+e*.1,t*.7,t*.6),2)
        return Image.fromarray(np.rint(rgb.clip(0,1)*255).astype(np.uint8)),False

    def run_compare(self,**kwargs):
        return compare_samples(self.dataset,self.config,self.identity,self.output,self.predictor,**kwargs)

    def test_all_three_outputs_conditions_and_provenance_are_recorded(self):
        page=self.run_compare();self.assertTrue(page.is_file());self.assertEqual(len(self.calls),3)
        records=json.loads((self.output/'records.json').read_text())
        self.assertEqual(set(records),{v+'/fixture' for v in VARIANTS})
        for x,p,q in self.calls:
            self.assertTrue(torch.equal(p,self.dataset[0]['p']));self.assertTrue(torch.equal(q,self.dataset[0]['q']))
        self.assertEqual(json.loads((self.output/'status.json').read_text())['phase'],'completed')

    def test_completed_resume_verifies_outputs_without_generation(self):
        self.run_compare();self.calls.clear();self.run_compare(resume=True)
        self.assertEqual(self.calls,[])

    def test_existing_output_is_preserved_and_changed_recipe_cannot_resume(self):
        self.run_compare()
        with self.assertRaises(ValueError):self.run_compare()
        self.config=dict(height=64,width=64)
        with self.assertRaises(ValueError):self.run_compare(resume=True)

    def test_tampered_RGB_or_condition_maps_are_not_silently_replaced(self):
        self.run_compare();path=self.output/'maps/fixture.npz';path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.run_compare(resume=True)
        self.assertEqual(path.read_bytes(),b'corrupt')

    def test_modified_source_condition_is_detected_before_predictor(self):
        (self.dataset.folder/'maps/fixture.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.run_compare()
        self.assertFalse(self.calls)

    def test_inference_failure_does_not_claim_completed(self):
        def fail(*args):raise FloatingPointError('forced numerical error')
        with self.assertRaises(FloatingPointError):compare_samples(self.dataset,self.config,self.identity,self.output,fail)
        self.assertEqual(json.loads((self.output/'status.json').read_text())['phase'],'failed')

    def test_semantic_chroma_statistics_do_not_call_G_a_quality_label(self):
        data=self.dataset[0];image,_=self.predictor(data['x'],data['p'],data['q'])
        metrics=rgb_metrics(image,data)
        self.assertTrue(np.isfinite(metrics['source_edge_proxy']))
        self.assertGreater(metrics['semantic_pair_weight'],0)
        self.assertTrue(0<=metrics['mean_saturation']<=1)

    def test_check_only_with_missing_formal_run_creates_no_output(self):
        with self.assertRaises(FileNotFoundError):main(['compare','--backend','formal','--run-dir',str(self.root/'missing'),
            '--output-dir',str(self.output),'--check-only'])
        self.assertFalse(self.output.exists())

    def test_unsafe_or_duplicate_input_identities_are_rejected(self):
        path=self.dataset.folder/'manifest.json';m=json.loads(path.read_text());m['frames'].append(copy.deepcopy(m['frames'][0]));write_json(path,m)
        write_json(self.dataset.folder/'status.json',dict(phase='completed',completed=2,total=2))
        with self.assertRaises(ValueError):ProbeDataset(self.dataset.folder)


if __name__=='__main__':unittest.main()
