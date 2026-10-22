"""Legacy checkpoint recovery: preserve update budgets and avoid unsafe relaunches."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
import run_learned_color
from options.train_options import TrainOptions
from util.training_progress import recovery_offset, recovery_position


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        name='independent_recovery'
        folder=root/'runs/checkpoints'/name
        folder.mkdir(parents=True)
        for net in ('G_A','G_B','D_A','D_B'):
            state={'weight':torch.ones(2,2)}
            if net=='G_A': state['color_loss_updates']=torch.tensor(10,dtype=torch.long)
            torch.save(state,folder/f'latest_net_{net}.pth')
        options=dict(n_epochs=2,n_epochs_decay=2,lr=.0002,seed=42,batch_size=1,
                     learned_color_mode='learned',learned_color_loss_weight=.01,
                     learned_color_loss_min_quality=.5,learned_color_loss_warmup_steps=1000,
                     reliability_flip=True,gate_semantic_conditions=True)
        (folder/'train_opt.txt').write_text('\n'.join(f'{key}: {value}' for key,value in options.items()))
        (folder/'color_train.json').write_text('{"initialization":"fresh"}')
        (root/'semantic.pth').write_text('not loaded')
        data=dict(dataset_id='independent',domains={
            'A':[dict(split='train') for _ in range(7)],
            'B':[dict(split='train') for _ in range(3)]+[dict(split='val')]})
        argv=['run_learned_color.py','--train','--resume','--name',name,'--gate-semantic',
              '--data-manifest',str(root/'data.json'),'--prior',str(root/'prior.json'),
              '--semantic-checkpoint',str(root/'semantic.pth'),'--epochs','4','--decay-epochs','2']
        return name,folder,data,argv

    def run_launcher(self, root, data, argv, active=()):
        captured=[]
        with patch.object(run_learned_color,'ROOT',root),patch.object(sys,'argv',argv), \
             patch('data.grouped_unaligned_dataset.validate_manifest',return_value=(data,root)), \
             patch('data.grouped_unaligned_dataset.validate_prior_partitions'), \
             patch('models.learned_color_prior.read_bank',return_value=({'classes':{'sky':{'status':'supported','loss_support':{}}}},'ab'*32)), \
             patch('torch.cuda.is_available',return_value=True), \
             patch.object(run_learned_color,'active_training',return_value=list(active)), \
             patch.object(run_learned_color.subprocess,'run',side_effect=lambda command,**kw:captured.append((command,kw))), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            run_learned_color.main()
        return captured

    def test_recovery_command_keeps_budget_and_backs_up_original_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); name,folder,data,argv=self.fixture(root)
            captured=self.run_launcher(root,data,argv)
            command,kwargs=captured[0]
            with patch.object(sys,'argv',['train.py',*command[3:]]):
                opt=TrainOptions().gather_options()
            self.assertTrue(opt.continue_train)
            self.assertEqual((opt.epoch,opt.epoch_count,opt.resume_total_iters),('latest',2,10))
            self.assertEqual((opt.n_epochs,opt.n_epochs_decay,opt.stop_after_epoch),(2,2,4))
            self.assertEqual(opt.num_threads,0)
            self.assertEqual(kwargs['env']['TMPDIR'],str((root/'runs/tmp'/name).resolve()))
            backups=list((root/'runs/recovery'/name).iterdir())
            self.assertEqual(len(backups),1)
            backup=backups[0]
            self.assertEqual((backup/'color_train.json').read_text(),'{"initialization":"fresh"}')
            plan=json.loads((backup/'recovery.json').read_text())
            self.assertEqual((plan['saved_updates'],plan['skip_positions'],plan['remaining_updates']),(10,3,18))
            for filename in plan['files']:
                self.assertEqual((folder/filename).read_bytes(),(backup/filename).read_bytes())

    def test_check_only_is_read_only_even_with_old_process_still_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); name,folder,data,argv=self.fixture(root)
            argv[1]='--check-only'
            self.assertEqual(self.run_launcher(root,data,argv,active=[123]),[])
            self.assertFalse((root/'runs/recovery').exists())
            self.assertFalse((root/'runs/tmp').exists())

    def test_active_process_changed_schedule_and_bad_weights_block_relaunch(self):
        for failure in ('active','schedule','nonfinite','missing','finished'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); name,folder,data,argv=self.fixture(root)
                active=[123] if failure=='active' else []
                if failure=='schedule':
                    argv[argv.index('--decay-epochs')+1]='1'
                if failure=='nonfinite':
                    torch.save({'weight':torch.tensor(float('nan'))},folder/'latest_net_D_B.pth')
                if failure=='missing': (folder/'latest_net_D_B.pth').unlink()
                if failure=='finished':
                    torch.save({'color_loss_updates':torch.tensor(28),'weight':torch.ones(2)},folder/'latest_net_G_A.pth')
                with self.assertRaises(SystemExit): self.run_launcher(root,data,argv,active=active)
                self.assertFalse((root/'runs/recovery').exists())

    def test_invalid_initialization_combinations_are_rejected(self):
        for extra in ('--from-scratch','--test-only'):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); name,folder,data,argv=self.fixture(root)
                with self.assertRaises(SystemExit): self.run_launcher(root,data,argv+[extra])

    def test_cursor_and_incompatible_training_modes(self):
        self.assertEqual(recovery_position(10,7),(2,3))
        self.assertEqual(recovery_position(14,7),(3,0))
        self.assertEqual(recovery_offset(10,7,2,1,True),3)
        for arguments in ((-1,7,2,1,True),(10,7,1,1,True),(10,7,2,2,True),
                          (10,7,2,1,False),(10,7,2,1,True,2)):
            with self.subTest(arguments=arguments),self.assertRaises(ValueError):
                recovery_offset(*arguments)

    def test_actual_training_loop_updates_only_remaining_positions(self):
        # Execute train.py against fake data/model, so the assertions cover the
        # real skip branch and save/end conditions, not a duplicate training loop.
        class Dataset:
            def __len__(self): return 7
            def __iter__(self): return iter(range(7))
            def set_epoch(self,epoch): pass
        class Model:
            def __init__(self): self.updates=10; self.values=[]; self.saved=[]
            def setup(self,opt): pass
            def set_input(self,value): self.values.append(value)
            def optimize_parameters(self): self.updates+=1
            def get_current_losses(self): return {}
            def update_learning_rate(self): pass
            def save_networks(self,suffix): self.saved.append((suffix,self.updates))
        opt=SimpleNamespace(stop_after_epoch=4,epoch_count=2,n_epochs=2,n_epochs_decay=2,
            seed=None,resume_total_iters=10,batch_size=1,continue_train=True,
            print_freq=4,display_freq=1000,save_latest_freq=1000,save_epoch_freq=5)
        model=Model()
        with patch('options.train_options.TrainOptions') as options, \
             patch('data.create_dataset',return_value=Dataset()),patch('models.create_model',return_value=model), \
             patch('util.visualizer.Visualizer',return_value=MagicMock()), \
             patch('util.util.init_ddp',return_value=torch.device('cpu')),patch('util.util.cleanup_ddp'), \
             contextlib.redirect_stdout(io.StringIO()):
            options.return_value.parse.return_value=opt
            runpy.run_path(str(ROOT/'pytorch-CycleGAN-and-pix2pix/train.py'),run_name='__main__')
        self.assertEqual(model.values,list(range(3,7))+list(range(7))*2)
        self.assertEqual(model.updates,28)
        self.assertEqual(model.saved,[('latest',28),(4,28)])

    def test_learning_rate_matches_original_schedule_after_recovery(self):
        from models.networks import get_scheduler
        def rates(start):
            optimizer=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=.0002)
            opt=SimpleNamespace(lr_policy='linear',epoch_count=start,n_epochs=2,n_epochs_decay=2)
            scheduler=get_scheduler(optimizer,opt)
            result=[]
            for epoch in range(start,5):
                result.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                scheduler.step()
            return result
        full=rates(1)
        for start in (2,3,4):
            self.assertEqual(rates(start),full[start-1:])


if __name__=='__main__': unittest.main()
