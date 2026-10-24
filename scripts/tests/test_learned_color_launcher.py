"""Parse the actual generated export command; launch no GPU job."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
import run_learned_color
from options.test_options import TestOptions
from options.train_options import TrainOptions


class LauncherTests(unittest.TestCase):
    def test_validation_export_counts_manifest_tir_and_parses_test_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); semantic=root/'semantic.pth'; semantic.write_text('not loaded')
            manifest=dict(dataset_id='independent',domains={'A':[], 'B':[dict(split='val') for _ in range(7)]})
            argv=['run_learned_color.py','--test-only','--data-manifest',str(root/'data.json'),
                  '--prior',str(root/'prior.json'),'--semantic-checkpoint',str(semantic),
                  '--name','launcher_test_uncreated','--phase','val','--gate-semantic','--color-loss-weight','0']
            captured=[]
            with patch.object(sys,'argv',argv), \
                 patch('data.grouped_unaligned_dataset.validate_manifest',return_value=(manifest,root)), \
                 patch('data.grouped_unaligned_dataset.validate_prior_partitions'), \
                 patch('models.learned_color_prior.read_bank',return_value=({'classes':{'sky':{'status':'supported'}}},'ab'*32)), \
                 patch('torch.cuda.is_available',return_value=True), \
                 patch.object(run_learned_color.subprocess,'run',side_effect=lambda command,**kw:captured.append(command)):
                run_learned_color.main()
            command=captured[0]
            self.assertEqual(command[command.index('--num_test')+1],'7')
            self.assertNotIn('--seed',command)
            with patch.object(sys,'argv',['test.py',*command[3:]]):
                options=TestOptions().gather_options()
            self.assertEqual(options.dataset_mode,'grouped_unaligned')
            self.assertEqual(options.model,'learned_color_cycle_gan')
            self.assertTrue(options.gate_semantic_conditions)
            self.assertEqual(options.learned_color_loss_weight,0)

    def test_tiny_loss_train_and_export_parse_with_identical_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); semantic=root/'semantic.pth'; semantic.write_text('not loaded')
            manifest=dict(dataset_id='independent',domains={'A':[], 'B':[dict(split='val')]})
            recipes=[]
            for variant,action,option_class in [(v,a,c) for v in ('selective_v1','clarity_v1','recall_v1')
                                               for a,c in [('--train',TrainOptions),('--test-only',TestOptions)]]:
                argv=['run_learned_color.py',action,'--data-manifest',str(root/'data.json'),
                      '--prior',str(root/'prior.json'),'--semantic-checkpoint',str(semantic),
                      '--name','micro_loss_uncreated','--color-loss-weight','.01','--color-loss-warmup-steps','1000',
                      '--contour-variant',variant]
                captured=[]
                with patch.object(sys,'argv',argv), \
                     patch('data.grouped_unaligned_dataset.validate_manifest',return_value=(manifest,root)), \
                     patch('data.grouped_unaligned_dataset.validate_prior_partitions'), \
                     patch('models.learned_color_prior.read_bank',return_value=({'classes':{'sky':{'status':'supported','loss_support':{}}}},'ab'*32)), \
                     patch('torch.cuda.is_available',return_value=True), \
                     patch.object(run_learned_color.subprocess,'run',side_effect=lambda command,**kw:captured.append(command)):
                    run_learned_color.main()
                with patch.object(sys,'argv',['entrypoint.py',*captured[0][3:]]):
                    options=option_class().gather_options()
                recipes.append((options.learned_color_loss_weight,options.learned_color_loss_min_quality,
                                options.learned_color_loss_warmup_steps,options.contour_variant))
            self.assertEqual(recipes,[(.01,.5,1000,v) for v in ('selective_v1','clarity_v1','recall_v1') for _ in range(2)])

    def test_scratch_defaults_generate_full_40_plus_40_schedule_without_loading_gans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); semantic=root/'semantic.pth'; semantic.write_text('not loaded')
            manifest=dict(dataset_id='independent',domains={'A':[], 'B':[dict(split='val')]})
            argv=['run_learned_color.py','--train','--from-scratch','--data-manifest',str(root/'data.json'),
                  '--prior',str(root/'prior.json'),'--semantic-checkpoint',str(semantic),'--name','scratch_test_uncreated']
            captured=[]
            with patch.object(sys,'argv',argv), \
                 patch('data.grouped_unaligned_dataset.validate_manifest',return_value=(manifest,root)), \
                 patch('data.grouped_unaligned_dataset.validate_prior_partitions'), \
                 patch('models.learned_color_prior.read_bank',return_value=({'classes':{'sky':{'status':'supported','loss_support':{}}}},'ab'*32)), \
                 patch('torch.cuda.is_available',return_value=True), \
                 patch.object(run_learned_color.subprocess,'run',side_effect=lambda command,**kw:captured.append(command)):
                run_learned_color.main()
            with patch.object(sys,'argv',['train.py',*captured[0][3:]]):
                options=TrainOptions().gather_options()
            self.assertEqual((options.n_epochs,options.n_epochs_decay,options.stop_after_epoch),(40,40,80))
            self.assertEqual(options.lr,.0002)
            self.assertEqual(options.save_epoch_freq,5)
            self.assertEqual(options.epoch_count,1)
            self.assertFalse(options.continue_train)
            self.assertEqual(options.contour_variant,'joint512')
            self.assertIsNone(options.init_condition_dir)
            self.assertIsNone(options.init_semantic_dir)

    def test_scratch_rejects_warmstart_and_invalid_decay_before_loading_data(self):
        base=['run_learned_color.py','--train','--from-scratch','--data-manifest','unused',
              '--prior','unused','--semantic-checkpoint','unused','--name','scratch_test_uncreated']
        for extra in (['--init-condition-dir','old_weights'],['--init-contour-dir','old_weights'],['--epochs','80','--decay-epochs','80']):
            with self.subTest(extra=extra),patch.object(sys,'argv',base+extra):
                with self.assertRaises(SystemExit): run_learned_color.main()


if __name__=='__main__': unittest.main()
