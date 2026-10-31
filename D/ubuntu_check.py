"""Verify an extracted Ubuntu source/assets upload without constructing GPU models."""

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path

from .common import ROOT, read_json, sha256, select_sources
from .package_assets import ASSET_DIR


def verify_manifest(path, expected_format):
    manifest=read_json(path)
    if manifest.get('format')!=expected_format:raise ValueError('Unknown upload manifest: '+str(path))
    for relative,digest in manifest['files'].items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT.resolve()) or sha256(source)!=digest:
            raise ValueError('Changed or missing upload file: '+relative)
    print(f'[VERIFY] {len(manifest["files"])} files: {path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args()
    verify_manifest(ROOT/'D/source_manifest.json','D_source_upload_v1')
    if args.source_only:return
    verify_manifest(ROOT/ASSET_DIR/'asset_manifest.json','D_ubuntu_offline_assets_v1')
    for name in ('torch','torchvision','diffusers','transformers','accelerate','safetensors','omegaconf',
                 'scipy','skimage','torchmetrics','torch_fidelity','matplotlib','filelock'):
        module=importlib.import_module(name)
        print(f'[DEPENDENCY] {name}: {getattr(module,"__version__","available")}',flush=True)
    import torch
    from .unpaired.semantics import default_semantic_checkpoint,legacy_models
    checkpoint=default_semantic_checkpoint()
    if not checkpoint.is_file():raise FileNotFoundError('Reuse/upload the complete FMB checkpoint: '+str(checkpoint))
    print(f'[FMB] {checkpoint} | SHA256 {sha256(checkpoint)}',flush=True)
    manifest=ROOT/ASSET_DIR/'cycle_manifest.json';prior=ROOT/ASSET_DIR/'rgb_prior_seed42_loss_v1.json'
    select_sources(manifest,limit=1)
    data=read_json(manifest)
    legacy_models()
    from data.grouped_unaligned_dataset import validate_prior_partitions
    from models.learned_color_prior import LearnedChromaPrior
    from models.semantic_edges import FMB_CLASS_NAMES
    validate_prior_partitions(data,manifest.parent,prior)
    LearnedChromaPrior(prior,FMB_CLASS_NAMES,require_loss_support=True)
    print('[SPLITS] '+json.dumps({k:dict(Counter(r['split'] for r in rows)) for k,rows in data['domains'].items()}),flush=True)
    # Catch a missing FID weight before a multi-hour training run reaches validation.
    actual=Path(torch.hub.get_dir())/'checkpoints/weights-inception-2015-12-05-6726825d.pth'
    expected=ROOT/'runs/D/torch_cache/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
    if actual.resolve()!=expected.resolve():raise ValueError('Set TORCH_HOME='+str(expected.parent.parent.parent))
    if not actual.is_file():raise FileNotFoundError(actual)
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric=FrechetInceptionDistance(feature=2048,normalize=True)
    del metric
    print('[METRIC] FID Inception loaded from the local Torch cache on CPU',flush=True)
    print(f'[CUDA] {torch.cuda.is_available()}',flush=True)
    if args.require_cuda and not torch.cuda.is_available():raise RuntimeError('Use the CUDA-enabled D environment')
    if torch.cuda.is_available():print('[GPU] '+torch.cuda.get_device_name(0),flush=True)
    print('[READY] Source/assets/data/prior checks passed. GPU model loading and training are separate checks.',flush=True)


if __name__=='__main__':main()
