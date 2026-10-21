#!/usr/bin/env python3
"""Real-weight one-pair implementation smoke, never a color-quality claim."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pytorch-CycleGAN-and-pix2pix'))
from check_semantic_color import image_tensor
from models import create_model
from options.train_options import TrainOptions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True); parser.add_argument('--semantic-checkpoint',type=Path,required=True)
    parser.add_argument('--tir',type=Path,required=True); parser.add_argument('--rgb',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--init-contour-dir',type=Path); parser.add_argument('--init-epoch',default='80')
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--color-loss-weight',type=float,default=.01)
    parser.add_argument('--color-loss-min-quality',type=float,default=.5)
    parser.add_argument('--color-loss-warmup-steps',type=int,default=1000)
    args=parser.parse_args()
    if args.output_dir.exists() or args.size<32 or args.size%4:
        parser.error('New output directory and a crop >=32 divisible by four required')
    argv=['train.py','--model','learned_color_cycle_gan','--dataroot',str(args.rgb.parent),
          '--name',args.output_dir.name,'--checkpoints_dir',str(args.output_dir.parent),
          '--semantic_checkpoint',str(args.semantic_checkpoint),'--learned_color_prior',str(args.prior),
          '--direction','BtoA','--input_nc','1','--output_nc','3','--netG','resnet_9blocks',
          '--ngf','64','--netD','pixel','--ndf','8','--norm','instance','--no_dropout',
          '--reliability_flip','--learned_color_dropout','0','--seed','42',
          '--learned_color_loss_weight',str(args.color_loss_weight),
          '--learned_color_loss_min_quality',str(args.color_loss_min_quality),
          '--learned_color_loss_warmup_steps',str(args.color_loss_warmup_steps)]
    if args.init_contour_dir: argv+=['--init_semantic_dir',str(args.init_contour_dir),'--init_semantic_epoch',args.init_epoch]
    original=sys.argv
    try:
        sys.argv=argv; opt=TrainOptions().gather_options()
    finally: sys.argv=original
    opt.isTrain=True; opt.device=torch.device('cpu')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(2); torch.manual_seed(42)
    started=time.perf_counter()
    model=create_model(opt); model.setup(opt)
    tir=image_tensor(args.tir,args.size,1); rgb=image_tensor(args.rgb,args.size,3)
    data=dict(B=tir,A=rgb,B_paths=[str(args.tir)],A_paths=[str(args.rgb)])
    model.set_input(data); model.eval(); model.test()
    g=model.netG_A
    with torch.no_grad():
        difference=float((model.fake_B-g.semantic(tir)).abs().max())
        if difference!=0: raise RuntimeError('Added zero-initialized branch changed warm-start output')
    color_gradient=0.
    if args.color_loss_weight>0:
        model.forward()
        raw,coverage=g.learned_prior.chroma_loss(model.fake_B,model.color_aux['probabilities'],
                                                model.color_aux['quality'],args.color_loss_min_quality)
        gradient=torch.autograd.grad(raw,model.fake_B)[0]
        color_gradient=float(gradient.abs().sum())
        if not torch.isfinite(gradient).all() or color_gradient<=0 or float(coverage)<=0:
            raise RuntimeError('Real sample has no active finite color gradient; inspect Q, support and sampled image')
    for name in model.model_names: getattr(model,'net'+name).train()
    losses=[]
    for _ in range(2):
        model.set_input(data); model.optimize_parameters()
        losses.append({key:float(getattr(model,'loss_'+key).detach()) if torch.is_tensor(getattr(model,'loss_'+key))
                       else float(getattr(model,'loss_'+key)) for key in model.loss_names})
    gradient=sum(float(p.grad.abs().sum()) for p in g.learned_projections.parameters() if p.grad is not None)
    encoder_gradient=sum(float(p.grad.abs().sum()) for p in g.learned_encoder.parameters() if p.grad is not None)
    if gradient<=0 or encoder_gradient<=0 or not all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in losses):
        raise RuntimeError('Missing conditioning gradient or nonfinite loss')
    if g.semantic_prior.model.training or any(p.grad is not None or p.requires_grad for p in g.semantic_prior.model.parameters()):
        raise RuntimeError('Semantic teacher not frozen')
    report=dict(evidence='real RGB/TIR, real frozen semantic weights and learned bank; two-step implementation smoke only',
        crop_size=args.size,generator='resnet9/ngf64',discriminator='small pixel/ndf8; smoke only',
        initial_max_difference=difference,projection_gradient_l1=gradient,encoder_gradient_l1=encoder_gradient,
        raw_color_gradient_l1=color_gradient,color_loss_weight=args.color_loss_weight,
        frozen_semantic=True,losses=losses,prior=g.learned_prior.metadata,
        quality_mean=float(model.color_aux['quality'].mean()),elapsed_seconds=time.perf_counter()-started,
        limits='No full RGB teacher cache, formal training, improved colors, lack of overfitting or external generalization established.')
    (args.output_dir/'smoke_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
