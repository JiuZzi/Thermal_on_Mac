#!/usr/bin/env python3
"""CSV adapter: one shared group split for GAN domains and the RGB prior."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import random

from color_prior_learning.data import digest,write_json


def prepare(args):
    with args.csv.open(newline='',encoding='utf-8-sig') as handle:
        records=list(csv.DictReader(handle))
    train_roles={args.train_a_role:'A',args.train_b_role:'B'}
    test_roles={role:domain for role,domain in ((args.test_a_role,'A'),(args.test_b_role,'B')) if role}
    if len(train_roles)!=2 or len(set(train_roles)|set(test_roles))!=len(train_roles)+len(test_roles):
        raise ValueError('Source domain/role values must be distinct')
    selected=[row for row in records if row[args.role_column] in set(train_roles)|set(test_roles)]
    if not selected or any(not row[args.group_column] for row in selected):
        raise ValueError('Empty selection or missing scene/video grouping')
    train_groups=sorted({row[args.group_column] for row in selected if row[args.role_column] in train_roles})
    test_groups={row[args.group_column] for row in selected if row[args.role_column] in test_roles}
    if set(train_groups)&test_groups:
        raise ValueError('Original training and test groups overlap')
    if not 0<args.val_fraction<.5 or len(train_groups)<4:
        raise ValueError('Need independent groups and validation fraction in (0,.5)')
    random.Random(args.seed).shuffle(train_groups)
    held=set(train_groups[:max(2,round(len(train_groups)*args.val_fraction))])
    output=args.output_dir.resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; partitions are preserved')
    image_root=args.image_root.resolve()
    domains={'A':[],'B':[]}; rgb=[]
    for row in sorted(selected,key=lambda r:(r[args.role_column],r[args.image_column])):
        role=row[args.role_column]; group=row[args.group_column]
        path=image_root/role/row[args.image_column] if args.role_is_subdir else image_root/row[args.image_column]
        path=path.resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError('Invalid source image')
        domain=train_roles.get(role,test_roles.get(role))
        split='test' if role in test_roles else ('val' if group in held else 'train')
        record=dict(image=os.path.relpath(path,output),group=group,split=split)
        domains[domain].append(record)
        if domain=='A':
            identity=hashlib.sha256(record['image'].encode()).hexdigest()[:20]
            rgb.append(dict(record,id=identity,label=f'labels/{identity}.png'))
    for domain in domains:
        for split in ('train','val'):
            if not any(r['split']==split for r in domains[domain]):
                raise ValueError('This seed leaves a modality without training/validation; choose a split before fitting')
    common=dict(dataset_id=args.dataset_id,seed=args.seed,group_definition=args.group_column,source_csv_sha256=digest(args.csv))
    write_json(output/'cycle_manifest.json',dict(common,format='grouped_unaligned_v1',domains=domains))
    write_json(output/'rgb_manifest.json',dict(common,format='rgb_chroma_manifest_v1',class_names=args.classes,images=rgb))
    print('Wrote grouped manifests:',output)
    for domain,rows in domains.items():
        print(domain,{split:len([r for r in rows if r['split']==split]) for split in ('train','val','test')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True); parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--dataset-id',required=True)
    parser.add_argument('--image-column',default='image'); parser.add_argument('--group-column',default='group')
    parser.add_argument('--role-column',default='split'); parser.add_argument('--train-a-role',default='trainA')
    parser.add_argument('--train-b-role',default='trainB'); parser.add_argument('--test-a-role'); parser.add_argument('--test-b-role')
    parser.add_argument('--role-is-subdir',action='store_true'); parser.add_argument('--classes',nargs='+',required=True)
    parser.add_argument('--val-fraction',type=float,default=.2); parser.add_argument('--seed',type=int,default=42)
    prepare(parser.parse_args())
