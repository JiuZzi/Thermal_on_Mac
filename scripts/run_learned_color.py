#!/usr/bin/env python3
"""Explicit grouped training/export; no FLIR image counts or color constants."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def active_training(name):
    """Check visible Linux training processes; never terminate processes here."""
    processes=Path('/proc')
    if not processes.is_dir(): return []
    active=[]
    for entry in processes.iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
        try:
            argv=entry.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError,ProcessLookupError):
            continue
        matches_name=any(arg=='--name' and index+1<len(argv) and argv[index+1]==name
                         or arg=='--name='+name for index,arg in enumerate(argv))
        if matches_name and any(Path(arg).name in ('train.py','run_learned_color.py') for arg in argv):
            active.append(int(entry.name))
    return active


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def inspect_recovery(folder, suffix, data, args):
    import torch
    from util.training_progress import recovery_position
    epoch_size=max(sum(row['split']=='train' for row in data['domains'][domain]) for domain in ('A','B'))
    paths={name:folder/f'{suffix}_net_{name}.pth' for name in ('G_A','G_B','D_A','D_B')}
    updates=None
    for name,path in paths.items():
        state=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not state or any(not torch.is_tensor(value) for value in state.values()):
            raise ValueError(f'{name}: expected a tensor state dictionary')
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f'{name}: non-finite checkpoint tensor')
        if name=='G_A':
            counter=state.get('color_loss_updates')
            if counter is None or counter.numel()!=1 or counter.dtype!=torch.long:
                raise ValueError('Checkpoint lacks a trustworthy color_loss_updates counter; do not guess the saved step')
            updates=int(counter)
        del state
    epoch,offset=recovery_position(updates,epoch_size)
    if epoch>args.epochs: raise ValueError('Saved updates already reach the requested total epoch budget')
    options={}
    for line in (folder/'train_opt.txt').read_text().splitlines():
        if ':' in line:
            key,value=line.strip().split(':',1)
            options[key.strip()]=value.split('[default:',1)[0].strip()
    expected=dict(n_epochs=args.epochs-args.decay_epochs,n_epochs_decay=args.decay_epochs,
                  lr=args.lr,seed=args.seed,batch_size=1,learned_color_mode=args.mode,
                  learned_color_loss_weight=args.color_loss_weight,
                  learned_color_loss_min_quality=args.color_loss_min_quality,
                  learned_color_loss_warmup_steps=args.color_loss_warmup_steps,
                  reliability_flip=not args.no_reliability_flip,gate_semantic_conditions=args.gate_semantic)
    for key,value in expected.items():
        if options.get(key)!=str(value):
            raise ValueError(f'Recovery must preserve {key}: saved {options.get(key)!r}, requested {value!r}')
    return dict(saved_updates=updates,epoch_size=epoch_size,epoch_count=epoch,skip_positions=offset,
                total_updates=args.epochs*epoch_size,remaining_updates=args.epochs*epoch_size-updates,
                restored='G_A/G_B/D_A/D_B weights and color-loss update counter',
                reset=['Adam state','replay pool','RNG and sampler state'],
                files={path.name:file_sha(path) for path in paths.values()})


def backup_recovery(folder, plan, name):
    destination=ROOT/'runs/recovery'/name
    destination.mkdir(parents=True,exist_ok=True)
    backup=Path(tempfile.mkdtemp(prefix=f"updates_{plan['saved_updates']}_",dir=destination))
    for filename,digest in plan['files'].items():
        shutil.copy2(folder/filename,backup/filename)
        if file_sha(backup/filename)!=digest:
            raise ValueError('Checkpoint changed during backup; stop the old trainer before recovering')
    for filename in ('train_opt.txt','color_train.json'):
        if (folder/filename).is_file(): shutil.copy2(folder/filename,backup/filename)
    (backup/'recovery.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    return backup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-only',action='store_true'); action.add_argument('--train',action='store_true')
    action.add_argument('--test-only',action='store_true')
    parser.add_argument('--data-manifest',type=Path,required=True); parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--semantic-checkpoint',type=Path,required=True); parser.add_argument('--name',required=True)
    parser.add_argument('--mode',choices=('learned','zero_tokens'),default='learned')
    parser.add_argument('--from-scratch',action='store_true',help='Fresh GANs; defaults to 80 epochs, half decay, lr 0.0002')
    parser.add_argument('--resume',action='store_true',help='Recover this existing experiment from weight/update-count checkpoints')
    parser.add_argument('--resume-epoch',default='latest')
    parser.add_argument('--init-condition-dir',type=Path); parser.add_argument('--init-condition-epoch',default='5')
    parser.add_argument('--init-contour-dir',type=Path); parser.add_argument('--init-contour-epoch',default='80')
    parser.add_argument('--epochs',type=int,help='Total epochs including decay; defaults: scratch 80, otherwise 5')
    parser.add_argument('--decay-epochs',type=int,help='Final epochs with linear LR decay; scratch defaults to half the total')
    parser.add_argument('--save-every',type=int,help='Numbered checkpoint interval; scratch defaults 5, otherwise 1')
    parser.add_argument('--epoch',default='5')
    parser.add_argument('--phase',choices=('val','test'),default='val')
    parser.add_argument('--feature-override',choices=('normal','off','shuffle'),default='normal')
    parser.add_argument('--lr',type=float,help='Defaults: scratch 0.0002, otherwise 0.00002'); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--color-loss-weight',type=float,default=.01,help='0 disables the learned GMM loss')
    parser.add_argument('--color-loss-min-quality',type=float,default=.5)
    parser.add_argument('--color-loss-warmup-steps',type=int,default=1000)
    parser.add_argument('--workers',type=int); parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--no-reliability-flip',action='store_true'); parser.add_argument('--gate-semantic',action='store_true')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'runs/results')
    args=parser.parse_args()
    full_run=args.from_scratch or args.resume
    if args.epochs is None: args.epochs=80 if full_run else 5
    if args.decay_epochs is None: args.decay_epochs=args.epochs//2 if full_run else 0
    if args.lr is None: args.lr=.0002 if full_run else .00002
    if args.save_every is None: args.save_every=5 if full_run else 1
    if args.workers is None: args.workers=0 if args.resume else 4
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.name) or args.name in ('.','..'):
        parser.error('Use one safe experiment directory name')
    if (args.epochs<1 or not 0<args.lr<float('inf') or args.workers<0
            or not 0<=args.decay_epochs<args.epochs or args.save_every<1):
        parser.error('Invalid training parameters')
    if args.from_scratch and (args.init_condition_dir or args.init_contour_dir or args.test_only):
        parser.error('--from-scratch is for training/check-only and cannot load old GAN generators')
    if args.resume and (args.from_scratch or args.init_condition_dir or args.init_contour_dir or args.test_only):
        parser.error('--resume cannot initialize fresh networks, warm-start generators, or export tests')
    if not re.fullmatch(r'(?:[0-9]+|latest)',args.resume_epoch): parser.error('Invalid recovery checkpoint suffix')
    if (not math.isfinite(args.color_loss_weight) or args.color_loss_weight<0
            or not math.isfinite(args.color_loss_min_quality) or not 0<args.color_loss_min_quality<=1
            or args.color_loss_warmup_steps<0):
        parser.error('Invalid color loss weight, reliability threshold or warmup')
    if args.init_condition_dir and args.init_contour_dir:
        parser.error('Use one explicit warm-start source')
    for folder,epoch in ((args.init_condition_dir,args.init_condition_epoch),(args.init_contour_dir,args.init_contour_epoch)):
        if folder:
            if not re.fullmatch(r'(?:[0-9]+|latest)',epoch): parser.error('Invalid initialization epoch')
            if not all((folder/f'{epoch}_net_{name}.pth').is_file() for name in ('G_A','G_B')):
                parser.error('Missing warm-start G_A/G_B files')
    if not re.fullmatch(r'(?:[0-9]+|latest)',args.epoch): parser.error('Invalid checkpoint epoch')
    sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
    from data.grouped_unaligned_dataset import validate_manifest,validate_prior_partitions
    from models.learned_color_prior import read_bank
    data,base=validate_manifest(args.data_manifest)
    bank,digest=read_bank(args.prior)
    if args.color_loss_weight>0 and any('loss_support' not in item for item in bank['classes'].values() if item['status']=='supported'):
        parser.error('Prior lacks training-only loss_support: refit with updated learn_color_prior.py fit, '
                     'or use --color-loss-weight 0 for the old conditioning-only control')
    validate_prior_partitions(data,base,args.prior)
    if not args.semantic_checkpoint.is_file():
        parser.error('Missing original TIR semantic checkpoint')
    checkpoints=ROOT/'runs/checkpoints'
    plan=None
    if args.resume:
        try:
            plan=inspect_recovery(checkpoints/args.name,args.resume_epoch,data,args)
        except (ValueError,OSError,RuntimeError) as error:
            parser.error(f'Cannot recover experiment: {error}')
    print(json.dumps(dict(dataset=data.get('dataset_id'),prior_sha256=digest,
                         support={k:v['status'] for k,v in bank['classes'].items()},
                         color_loss=dict(weight=args.color_loss_weight,min_quality=args.color_loss_min_quality,
                                         warmup_steps=args.color_loss_warmup_steps),
                         initialization='semantic GAN warm-start' if args.init_condition_dir else
                                        ('contour GAN warm-start' if args.init_contour_dir else
                                         ('recover experiment weights' if args.resume else
                                          ('load experiment checkpoint' if args.test_only else 'fresh GAN networks'))),
                         recovery=plan,
                         training_schedule=dict(total_epochs=args.epochs,constant_epochs=args.epochs-args.decay_epochs,
                                                decay_epochs=args.decay_epochs,learning_rate=args.lr,save_every=args.save_every),
                         note=('Old warm-start weights may have seen the new validation groups. '
                               if args.init_condition_dir or args.init_contour_dir else '')+
                              'RGB/TIR segmenters stay pretrained and frozen; external generalization is unverified.'),ensure_ascii=False,indent=2))
    if args.check_only:
        return
    if args.train and (checkpoints/args.name).exists() and not args.resume:
        parser.error('Preserve existing experiment; choose a new name')
    if args.resume and (running:=active_training(args.name)):
        parser.error(f'Existing training processes still running: {running}; stop this experiment before recovery')
    if args.device=='cpu':
        parser.error('Use CPU for check-only or check_learned_color.py; full launcher selects CUDA')
    if args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable; --check-only checks data/artifact identity without training')
    command=[sys.executable,'-u','train.py' if args.train else 'test.py',
             '--dataroot',str(base),'--dataset_mode','grouped_unaligned','--grouped_data_manifest',str(args.data_manifest.resolve()),
             '--model','learned_color_cycle_gan','--name',args.name,'--checkpoints_dir',str(checkpoints),
             '--semantic_checkpoint',str(args.semantic_checkpoint.resolve()),'--learned_color_prior',str(args.prior.resolve()),
             '--learned_color_mode',args.mode,'--condition_mode','soft_saliency','--fusion_mode','direct',
             '--direction','BtoA','--input_nc','1','--output_nc','3','--netG','resnet_9blocks',
             '--netD','basic','--ngf','64','--ndf','64','--norm','instance','--no_dropout',
             '--color_condition','semantic','--lambda_color_prior','0',
             '--learned_color_loss_weight',str(args.color_loss_weight),
             '--learned_color_loss_min_quality',str(args.color_loss_min_quality),
             '--learned_color_loss_warmup_steps',str(args.color_loss_warmup_steps)]
    if not args.no_reliability_flip: command+=['--reliability_flip']
    if args.gate_semantic: command+=['--gate_semantic_conditions']
    if args.train:
        if args.feature_override!='normal': parser.error('Perturbations are test diagnostics only')
        command+=['--seed',str(args.seed),'--phase','train','--preprocess','crop','--crop_size','256','--batch_size','1',
                  '--num_threads',str(args.workers),'--lr',str(args.lr),'--lambda_identity','0',
                  '--epoch_count',str(plan['epoch_count'] if plan else 1),'--lr_policy','linear',
                  '--n_epochs',str(args.epochs-args.decay_epochs),'--n_epochs_decay',str(args.decay_epochs),
                  '--stop_after_epoch',str(args.epochs),'--save_epoch_freq',str(args.save_every)]
        if plan:
            command+=['--continue_train','--epoch',args.resume_epoch,
                      '--resume_total_iters',str(plan['saved_updates'])]
        if args.init_condition_dir:
            command+=['--init_condition_dir',str(args.init_condition_dir.resolve()),'--init_condition_epoch',args.init_condition_epoch]
        if args.init_contour_dir:
            command+=['--init_semantic_dir',str(args.init_contour_dir.resolve()),'--init_semantic_epoch',args.init_contour_epoch]
    else:
        count=sum(r['split']==args.phase for r in data['domains']['B'])
        if not count: parser.error('No TIR images for requested evaluation role')
        if (args.results_dir/args.name/f'{args.phase}_{args.epoch}').exists():
            parser.error('Preserve existing export; choose another results directory')
        command+=['--phase',args.phase,'--epoch',args.epoch,'--results_dir',str(args.results_dir.resolve()),
                  '--color_feature_override',args.feature_override,'--num_test',str(count),'--preprocess','none','--no_flip','--eval']
    print(shlex.join(command),flush=True)
    extra={}
    if plan:
        temporary=ROOT/'runs/tmp'/args.name
        temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=temporary): pass
        backup=backup_recovery(checkpoints/args.name,plan,args.name)
        extra['env']=dict(os.environ,TMPDIR=str(temporary.resolve()))
        print(f'Recovery backup: {backup}; TMPDIR: {temporary}; data workers: {args.workers}',flush=True)
    subprocess.run(command,cwd=ROOT/'pytorch-CycleGAN-and-pix2pix',check=True,**extra)


if __name__=='__main__': main()
