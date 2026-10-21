"""Data leakage barriers and actual multimodal fitting, without RGB weights."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from color_prior_learning.data import balanced_groups,sample_manifest,write_json
from color_prior_learning.fit import fit_bank,fit_mixture,log_density,evaluate_bank

CONFIG=dict(components=[1,2,3],variance_floor=1.,min_train_groups=3,min_val_groups=2,
            min_group_pixels=16,image_cap=128,group_cap=256,seed=42,bootstrap_runs=2)


def dataset(root):
    rng=np.random.default_rng(112)
    rows=[]
    for index,role in enumerate(['train']*4+['val']*2+['test']*2):
        # Pixel data, not predeclared chroma centers, supplies every learned color.
        rgb=np.zeros((24,40,3),np.uint8)
        rgb[:,:20]=np.clip(rng.normal([70+index,120,50],5,(24,20,3)),0,255)
        rgb[:,20:]=np.clip(rng.normal([100,160,210-index],5,(24,20,3)),0,255)
        labels=np.zeros((24,40),np.uint8); labels[:,20:]=1
        Image.fromarray(rgb).save(root/f'{index}.png')
        Image.fromarray(labels).save(root/f'{index}_label.png')
        rows.append(dict(id=str(index),group=f'scene_{index}',split=role,image=f'{index}.png',label=f'{index}_label.png'))
    manifest=dict(format='rgb_chroma_manifest_v1',dataset_id='synthetic_independent_scenes',
                  class_names=['vegetation','sky','missing'],images=rows)
    write_json(root/'manifest.json',manifest)
    return root/'manifest.json',manifest


class PriorLearningTests(unittest.TestCase):
    def test_em_recovers_separated_modes_and_improves_unseen_density(self):
        rng=np.random.default_rng(7)
        x=np.concatenate([rng.normal([-25,20],2,(500,2)),rng.normal([30,-20],3,(500,2))])
        unseen=np.concatenate([rng.normal([-25,20],2,(200,2)),rng.normal([30,-20],3,(200,2))])
        one=fit_mixture(x,1,1.,4); two=fit_mixture(x,2,1.,4)
        self.assertGreater(log_density(unseen,two).mean(),log_density(unseen,one).mean()+1)
        np.testing.assert_allclose(two['means_ab'],[[-25,20],[30,-20]],atol=1)
        self.assertTrue(np.all(np.asarray(two['variances_ab'])>=1))

    def test_fit_does_not_read_test_pixels_and_marks_missing_class_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,manifest=dataset(root)
            bank=fit_bank(path,root/'bank.json',CONFIG)
            self.assertEqual(bank['classes']['missing']['status'],'unsupported')
            self.assertEqual({r['split'] for r in bank['provenance']['samples']},{'train','val'})
            # Test files may be absent: the fit must not open them.
            (root/'6.png').unlink(); (root/'7.png').unlink()
            other=fit_bank(path,root/'bank2.json',CONFIG)
            self.assertEqual(bank,other)
            self.assertEqual(bank['classes']['sky']['support']['val_groups'],2)
            _,pools,_=sample_manifest(path,CONFIG['image_cap'],CONFIG['group_cap'],CONFIG['seed'])
            _,training=balanced_groups(pools,'train','sky',CONFIG['min_group_pixels'])
            model=bank['classes']['sky']
            self.assertEqual(model['loss_support']['source'],'group_balanced_train_only')
            self.assertAlmostEqual(model['loss_support']['max_nll'],float(np.quantile(-log_density(training,model),.99)))

    def test_group_and_decoded_duplicate_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,manifest=dataset(root)
            changed=copy.deepcopy(manifest); changed['images'][4]['group']='scene_0'
            write_json(root/'bad_group.json',changed)
            with self.assertRaisesRegex(ValueError,'Group leaks'):
                sample_manifest(root/'bad_group.json')
            Image.open(root/'0.png').save(root/'duplicate.png')
            changed=copy.deepcopy(manifest); changed['images'][4]['image']='duplicate.png'
            write_json(root/'bad_pixel.json',changed)
            with self.assertRaisesRegex(ValueError,'duplicate leaks'):
                sample_manifest(root/'bad_pixel.json')

    def test_sampling_and_validation_selection_are_bounded_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,_=dataset(root)
            _,pools,_=sample_manifest(path,image_cap=17,group_cap=12)
            self.assertTrue(all(len(v)<=12 for v in pools.values()))
            bank=fit_bank(path,root/'bank.json',CONFIG)
            self.assertEqual(len(bank['classes']['sky']['bootstrap_validation_nll']),2)
            self.assertIn(bank['classes']['sky']['selected_components'],CONFIG['components'])
            with self.assertRaisesRegex(ValueError,'Preserve existing'):
                fit_bank(path,root/'bank.json',CONFIG)

    def test_independent_role_evaluation_and_false_test_reuse_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,manifest=dataset(root)
            fit_bank(path,root/'bank.json',CONFIG)
            result=evaluate_bank(root/'bank.json',path,root/'test_scores.json')
            self.assertEqual(set(result['scores']),{'vegetation','sky'})
            # Relabel selected validation frames as a supposed new test set.
            changed=copy.deepcopy(manifest)
            changed['images']=[dict(r,split='test') for r in manifest['images'] if r['split']=='val']
            write_json(root/'fake_external.json',changed)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                evaluate_bank(root/'bank.json',root/'fake_external.json',root/'must_not_exist.json')

    def test_rgb_teacher_adapter_restores_geometry_freezes_weights_and_skips_test(self):
        import torch
        import learn_color_prior
        class Inputs(dict):
            def to(self,device): return self
        class Processor:
            @classmethod
            def from_pretrained(cls,*args,**kwargs): return cls()
            def __call__(self,images,return_tensors):
                return Inputs(pixel_values=torch.from_numpy(np.asarray(images).copy()).permute(2,0,1)[None].float())
            def to_dict(self): return {'test_stub':True}
        class Teacher(torch.nn.Module):
            config=SimpleNamespace(id2label={0:'vegetation',1:'sky',2:'missing'})
            @classmethod
            def from_pretrained(cls,*args,**kwargs): return cls()
            def forward(self,pixel_values):
                if self.training: raise AssertionError('Teacher must stay in eval mode')
                logits=torch.zeros(1,3,6,10); logits[:,0]=20
                return SimpleNamespace(logits=logits)
        fake=ModuleType('transformers'); fake.AutoImageProcessor=Processor
        fake.AutoModelForSemanticSegmentation=Teacher; fake.__version__='stub'
        hub=ModuleType('huggingface_hub'); hub.snapshot_download=lambda *a,**k: None
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,_=dataset(root)
            args=SimpleNamespace(manifest=path,teacher=str(root),revision='local_test',output_dir=root/'teacher_cache',
                allow_download=False,device='cpu',cpu_threads=1,include_test=False,flip=True,min_probability=.9,
                min_margin=.2,min_agreement=.9,erosion=2)
            with patch.dict(sys.modules,{'transformers':fake,'huggingface_hub':hub}):
                learn_color_prior.cache(args)
            cached=json.loads((args.output_dir/'manifest.json').read_text())
            self.assertEqual(len(cached['images']),6)
            mask=np.asarray(Image.open(args.output_dir/cached['images'][0]['label']))
            self.assertEqual(mask.shape,(24,40))
            self.assertEqual(mask[12,20],0)
            self.assertEqual(mask[0,0],255)


if __name__=='__main__': unittest.main()
