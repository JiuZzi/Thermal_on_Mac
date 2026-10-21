"""Unpaired image loading from explicit, leakage-checked group partitions."""

import hashlib
import json
from pathlib import Path
import random

from PIL import Image

from .base_dataset import BaseDataset, get_transform


def validate_manifest(path):
    path=Path(path).resolve()
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format')!='grouped_unaligned_v1' or set(manifest.get('domains',{}))!={'A','B'}:
        raise ValueError('Expected grouped_unaligned_v1 with A/B domains')
    groups,paths,hashes={}, {}, {}
    for domain,rows in manifest['domains'].items():
        for row in rows:
            if not row.get('group') or row.get('split') not in ('train','val','test'):
                raise ValueError('Every image needs an explicit group and role')
            group=str(row['group']); role=row['split']
            if group in groups and groups[group]!=role:
                raise ValueError('Scene/video group leaks across roles, including across modalities')
            groups[group]=role
            image=(path.parent/row['image']).resolve()
            if not image.is_file() or image in paths:
                raise ValueError('Missing or repeated image path')
            paths[image]=role
            with Image.open(image) as handle:
                rgb=handle.convert('RGB')
                identity=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if identity in hashes and hashes[identity]!=role:
                raise ValueError('Decoded duplicate image leaks across roles')
            hashes[identity]=role
    return manifest,path.parent


def validate_prior_partitions(manifest,base,artifact):
    bank=json.loads(Path(artifact).read_text(encoding='utf-8'))
    source=bank.get('provenance',{})
    records=source.get('samples',[])
    for role in ('val','test'):
        forbidden=[r for r in records if role=='test' or r['split']=='train']
        groups={str(r['group']) for r in forbidden} if source.get('dataset_id')==manifest.get('dataset_id') else set()
        contents={r['rgb_content_sha256'] for r in forbidden}
        for domain,rows in manifest['domains'].items():
            for row in rows:
                if row['split']!=role: continue
                if str(row['group']) in groups:
                    raise ValueError('Color-prior fitting leaks into GAN validation/test groups')
                if domain=='A':
                    with Image.open(base/row['image']) as handle:
                        rgb=handle.convert('RGB')
                        identity=hashlib.sha256(str((rgb.height,rgb.width,3)).encode()+rgb.tobytes()).hexdigest()
                    if identity in contents:
                        raise ValueError('Color-prior RGB content overlaps GAN validation/test')


class GroupedUnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser,is_train):
        parser.add_argument('--grouped_data_manifest',required=True)
        return parser

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        manifest,base=validate_manifest(opt.grouped_data_manifest)
        if getattr(opt,'learned_color_prior',None):
            validate_prior_partitions(manifest,base,opt.learned_color_prior)
        self.A_paths=[str((base/r['image']).resolve()) for r in manifest['domains']['A'] if r['split']==opt.phase]
        self.B_paths=[str((base/r['image']).resolve()) for r in manifest['domains']['B'] if r['split']==opt.phase]
        self.A_paths.sort(); self.B_paths.sort()
        if not self.A_paths or not self.B_paths:
            raise ValueError(f'Missing RGB/TIR images for role {opt.phase}; no fallback to another role')
        self.A_size,self.B_size=len(self.A_paths),len(self.B_paths)
        if opt.direction!='BtoA' or opt.input_nc!=1 or opt.output_nc!=3:
            raise ValueError('Grouped thermal pipeline expects BtoA, 1 to 3 channels')
        self.transform_A=get_transform(opt,grayscale=False)
        self.transform_B=get_transform(opt,grayscale=True)

    def __getitem__(self,index):
        a=self.A_paths[index%self.A_size]
        j=index%self.B_size if self.opt.serial_batches else random.randint(0,self.B_size-1)
        b=self.B_paths[j]
        with Image.open(a) as handle: rgb=self.transform_A(handle.convert('RGB'))
        with Image.open(b) as handle: tir=self.transform_B(handle.convert('RGB'))
        return dict(A=rgb,B=tir,A_paths=a,B_paths=b)

    def __len__(self):
        return max(self.A_size,self.B_size)
