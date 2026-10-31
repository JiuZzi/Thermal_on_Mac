import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from D.common import write_json, sha256
from D.diagnostics.compare_edge_use import code_identity, preflight
from D.diagnostics.discover import discover


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.runs=self.root/'training';self.caches=self.root/'conditions'
        self.prior=self.root/'prior.json';write_json(self.prior,{'prior':'fixture'})
        self.source=dict(recipe={'size':[64,64]},source_manifest_sha256='source fixture')
        self.train=self.make_cache('train_and_val',self.source,4)
        self.val=self.make_cache('val_full',self.source,8)
        self.other=self.make_cache('other_recipe',dict(recipe={'size':[128,128]}),16)
        self.run=self.make_run('trained',step=123)

    def make_cache(self,name,identity,count):
        folder=self.caches/name;folder.mkdir(parents=True)
        frames=[dict(id=str(i),stem=str(i),domain='TIR',role='val') for i in range(count)]
        write_json(folder/'manifest.json',dict(format='D_unpaired_conditions_v1',identity=identity,frames=frames))
        write_json(folder/'status.json',dict(phase='completed',completed=count,total=count))
        return folder

    def make_run(self,name,step):
        folder=self.runs/name;target=folder/'checkpoints'/f'step_{step:06d}';target.mkdir(parents=True)
        run=dict(format='D_unpaired_run_v1',config=dict(width=64,height=64),data={'source':copy.deepcopy(self.source)},
            prior_sha256=sha256(self.prior),code_sha256=code_identity(),
            validation={'cache_manifest_sha256':sha256(self.val/'manifest.json')})
        write_json(folder/'run.json',run)
        names=('generator.safetensors','reverse.safetensors','critics.safetensors','training.pt')
        for name in names:(target/name).write_bytes(b'fixture: metadata check only')
        write_json(target/'checkpoint.json',dict(format='D_unpaired_checkpoint_v1',run_sha256=sha256(folder/'run.json'),
            step=step,files={name:sha256(target/name) for name in names}))
        write_json(folder/'latest.json',dict(checkpoint=f'checkpoints/step_{step:06d}',step=step))
        return folder

    def inventory(self,**kwargs):return discover(self.runs,self.caches,self.prior,**kwargs)

    def test_discovers_matching_validation_only_and_prefers_registered_cache(self):
        test_only=self.make_cache('test_only',self.source,2)
        m=json.loads((test_only/'manifest.json').read_text())
        for row in m['frames']:row['role']='test'
        write_json(test_only/'manifest.json',m)
        row=self.inventory()['runs'][0]
        self.assertEqual(row['latest_step'],123)
        self.assertEqual([r['path'] for r in row['conditions']],[str(self.val.resolve()),str(self.train.resolve())])
        self.assertTrue(row['prior_matches']);self.assertFalse(row['best_file_present'])

    def test_prior_mismatch_is_reported_without_touching_files(self):
        write_json(self.prior,{'prior':'different'})
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertFalse(self.inventory()['runs'][0]['prior_matches'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_incomplete_latest_is_reported_without_substituting_another_checkpoint(self):
        (self.run/'checkpoints/step_000123/generator.safetensors').unlink()
        row=self.inventory()['runs'][0]
        self.assertIsNone(row['latest_step']);self.assertIn('Incomplete',row['checkpoint_error'])

    def test_nonformal_and_incomplete_caches_are_excluded(self):
        (self.runs/'warmup').mkdir();write_json(self.runs/'warmup/run.json',dict(format='RGB_warmup'))
        write_json(self.val/'status.json',dict(phase='preparing',completed=7,total=8))
        rows=self.inventory()['runs'];self.assertEqual(len(rows),1)
        self.assertEqual([r['path'] for r in rows[0]['conditions']],[str(self.train.resolve())])

    def test_requested_unknown_run_fails_instead_of_choosing_a_different_run(self):
        with self.assertRaises(ValueError):self.inventory(run_dir=self.root/'missing')

    def test_formal_preflight_hashes_checkpoint_and_refuses_changed_inference_code(self):
        args=SimpleNamespace(backend='formal',run_dir=self.run,prior=self.prior,conditions=self.val,step=None,best=False)
        dataset,config,identity=preflight(args)
        self.assertEqual(len(dataset),8);self.assertEqual(identity['step'],123)
        (self.run/'checkpoints/step_000123/generator.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Checkpoint file changed'):preflight(args)
        m=json.loads((self.run/'run.json').read_text());m['code_sha256']['D/unpaired/model.py']='different'
        write_json(self.run/'run.json',m)
        with self.assertRaisesRegex(ValueError,'Inference source differs'):preflight(args)

    def test_formal_preflight_does_not_accept_a_different_condition_recipe(self):
        args=SimpleNamespace(backend='formal',run_dir=self.run,prior=self.prior,conditions=self.other,step=None,best=False)
        with self.assertRaisesRegex(ValueError,'cache must match'):preflight(args)


if __name__=='__main__':unittest.main()
