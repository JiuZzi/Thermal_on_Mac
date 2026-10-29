import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from D.common import read_json, write_json
from D.unpaired import checkpoint
from D.unpaired.engine import UnpairedEngine, NumericalFailure
from D.unpaired.protocol import config_from, IndexStream
from D.unpaired.train import run_updates, validate_recovery, LEGACY_NUMERICAL_FILES
from D.unpaired.tests import test_formal as formal


class FakeScaler:
    def __init__(self): self.factor = 1.
    def is_enabled(self): return True
    def get_scale(self): return self.factor
    def get_backoff_factor(self): return .5
    def scale(self, loss): return loss*self.factor
    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None: parameter.grad.div_(self.factor)
    def update(self, new_scale=None):
        if new_scale is not None: self.factor = new_scale
    def step(self, optimizer): optimizer.step()


class AtomicUpdateTests(unittest.TestCase):
    setUp = formal.FormalGeneratorTests.setUp
    tearDown = formal.FormalGeneratorTests.tearDown

    def make_engine(self):
        x,p,q = formal.conditions()
        class Builder:
            def synthetic(self,tir):
                return torch.cat(((tir+1)/2,x[:,1:2],x[:,2:3]),1),p,q
        class Teacher(nn.Module):
            def forward(self,rgb,side=256):
                return rgb.mean(1,keepdim=True).repeat(1,15,1,1).softmax(1)
        reverse = nn.Sequential(nn.Conv2d(3,1,3,padding=1),nn.Tanh())
        critics = nn.ModuleDict(dict(RGB=nn.Conv2d(3,1,1),TIR=nn.Conv2d(1,1,1)))
        config = copy.deepcopy(self.config);config['replay_every']=1
        engine = UnpairedEngine(self.model,reverse,critics,Builder(),Teacher(),config,device='cpu')
        sample = dict(x=x,p=p,q=q,rgb=torch.rand(1,3,64,64)*2-1)
        return engine,sample

    def test_amp_scale_can_fall_below_one_and_rejected_noise_is_reused(self):
        engine,sample = self.make_engine();engine.scaler=FakeScaler()
        before=engine.retry_state()
        with patch('torch.nn.utils.clip_grad_norm_',return_value=torch.tensor(float('inf'))):
            for expected in (.5,.25):
                applied,report=engine.update(sample,sample,0,10)
                self.assertFalse(applied);self.assertEqual(report['amp_scale_after'],expected)
                self.assertTrue(torch.equal(before['noise_rng'],engine.noise_rng.get_state()))
                self.assertTrue(torch.equal(before['patch_rng'],engine.patch_rng.get_state()))
                self.assertEqual(len(engine.rgb_pool.images),0)
        applied,report=engine.update(sample,sample,0,10,diagnostics=True)
        self.assertTrue(applied);self.assertEqual(report['retry_count'],2)

    def test_discriminator_failure_does_not_partially_step_generator(self):
        engine,sample=self.make_engine()
        modules=(engine.generator,engine.reverse,engine.critics)
        before=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in modules]
        pool_before=engine.retry_state()
        with patch('torch.nn.utils.clip_grad_norm_',side_effect=[torch.tensor(1.),torch.tensor(float('inf'))]):
            applied,report=engine.update(sample,sample,0,10)
        self.assertFalse(applied);self.assertEqual(report['stage'],'discriminator_gradients')
        for module,expected in zip(modules,before):
            self.assertTrue(all(torch.equal(v,expected[k]) for k,v in module.state_dict().items()))
        self.assertFalse(engine.opt_g.state);self.assertFalse(engine.opt_d.state)
        self.assertEqual(len(engine.rgb_pool.images),0)
        self.assertEqual(engine.rgb_pool.rng.getstate(),pool_before['rgb_pool']['rng'])


class ToyGenerator(nn.Linear):
    def __init__(self): super().__init__(1,1)
    def export_state(self): return {k:v.detach().clone() for k,v in self.state_dict().items()}
    def import_state(self,values): self.load_state_dict(values,strict=True)


def toy_engine():
    return UnpairedEngine(ToyGenerator(),nn.Linear(1,1),nn.ModuleDict(dict(D=nn.Linear(1,1))),
                          None,None,config_from(),device='cpu')


class RecoveryTests(unittest.TestCase):
    def test_rolling_snapshots_keep_two_and_archival_epochs_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);write_json(folder/'run.json',{'fixture':1})
            engine=toy_engine();streams=dict(TIR=IndexStream(3,'TIR'),RGB=IndexStream(4,'RGB'))
            archive=checkpoint.save(folder,5,engine,streams)
            for step in (6,7,8):
                streams['TIR'].next();streams['RGB'].next()
                checkpoint.save(folder,step,engine,streams,kind='recovery')
            self.assertTrue(archive.exists());self.assertEqual(len(list((folder/'recovery').iterdir())),2)
            location,step=checkpoint.locate(folder);self.assertEqual(step,8)
            target=toy_engine();restored=dict(TIR=IndexStream(3,'TIR'),RGB=IndexStream(4,'RGB'))
            self.assertEqual(checkpoint.restore(folder,location,target,restored),8)
            self.assertEqual(restored['TIR'].position,streams['TIR'].position)
            self.assertTrue(torch.equal(target.noise_rng.get_state(),engine.noise_rng.get_state()))
            self.assertTrue(torch.equal(target.generator.weight,engine.generator.weight))

    def test_emergency_save_retains_last_good_update_and_rolls_back_pending_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);write_json(folder/'run.json',{'fixture':1})
            engine=toy_engine();streams=dict(TIR=IndexStream(3,'TIR'),RGB=IndexStream(3,'RGB'))
            class Data:
                def __len__(self):return 3
                def __getitem__(self,index):return dict(frame=dict(filename=f'frame{index}'),x=torch.ones(1))
            def update(tir,rgb,step,total,diagnostics=False):
                torch.randn((2,),generator=engine.noise_rng)
                if step==1:
                    engine.phase='rejected'
                    raise NumericalFailure(dict(stage='forced_regression_failure',optimizers_unchanged=True))
                with torch.no_grad():engine.generator.weight.add_(.01)
                return True,dict(loss=torch.tensor(1.))
            engine.update=update
            args=SimpleNamespace(steps=3,verify_updates=0,log_every=1,save_every=3,recovery_every=100,preview_every=0)
            with self.assertRaises(NumericalFailure):
                run_updates(args,folder,engine,streams,Data(),Data(),0)
            self.assertEqual(read_json(folder/'status.json')['phase'],'failed')
            location,step=checkpoint.locate(folder);self.assertEqual(step,1)
            self.assertIn('emergency',location.parts)
            target=toy_engine();restored=dict(TIR=IndexStream(3,'TIR'),RGB=IndexStream(3,'RGB'))
            checkpoint.restore(folder,location,target,restored)
            self.assertEqual(restored['TIR'].position,1)
            expected=torch.Generator('cpu').manual_seed(42);torch.randn((2,),generator=expected)
            self.assertTrue(torch.equal(target.noise_rng.get_state(),expected.get_state()))

    def test_audited_full_state_migration_rejects_other_changes(self):
        original=dict(config={'seed':42},data={'manifest':'same'},prior_sha256='bank',versions={'torch':'2.6'},
                      initialization={'kind':'same'},schedule=dict(steps=27655,save_every=5531,log_every=10,preview_every=5531),
                      code_sha256={'unchanged.py':'same','D/unpaired/losses.py':LEGACY_NUMERICAL_FILES['D/unpaired/losses.py']},
                      format='D_unpaired_run_v1')
        fixed=copy.deepcopy(original);fixed['code_sha256']['D/unpaired/losses.py']='patched'
        validate_recovery(original,fixed)
        for key in ('config','data','prior_sha256'):
            altered=copy.deepcopy(fixed);altered[key]='changed'
            with self.assertRaises(ValueError):validate_recovery(original,altered)
        altered=copy.deepcopy(fixed);altered['schedule']['steps']=30000
        with self.assertRaises(ValueError):validate_recovery(original,altered)
        altered=copy.deepcopy(fixed);altered['code_sha256']['unchanged.py']='changed'
        with self.assertRaises(ValueError):validate_recovery(original,altered)


if __name__=='__main__':unittest.main()
