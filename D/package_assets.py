"""Bundle existing pinned models and frozen color/split artifacts for offline Ubuntu.

No downloads or training. Snapshot symlinks are deliberately stored as regular
files, so the extracted cache does not depend on the Mac's blob directory.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from .common import ROOT, DEFAULT_CACHE, sha256
from .prepare_rgb import TEACHER, REVISION


ASSET_DIR = 'runs/D/ubuntu_assets_20261030'


def collect(cache, rgb_snapshot, inception, prior_archive):
    config=json.loads((ROOT/'D/unpaired/config.json').read_text(encoding='utf-8'))
    files={}
    for kind in ('base','control'):
        relative=Path('runs/D/model_cache')/('models--'+config[kind+'_model'].replace('/','--'))/'snapshots'/config[kind+'_revision']
        source=Path(cache)/relative.relative_to('runs/D/model_cache')
        selected=[p for p in sorted(source.rglob('*')) if p.is_file() and
            (p.suffix in ('.json','.txt','.safetensors','.md') or p.name.startswith('LICENSE') or p.name=='NOTICE')]
        if not any(p.suffix=='.safetensors' for p in selected):raise FileNotFoundError('Missing local snapshot: '+str(source))
        for p in selected:files[(relative/p.relative_to(source)).as_posix()]=p
    rgb_relative=Path('runs/D/model_cache')/('models--'+TEACHER.replace('/','--'))/'snapshots'/REVISION
    for name in ('config.json','preprocessor_config.json','pytorch_model.bin','README.md'):
        source=Path(rgb_snapshot)/name
        if name=='README.md' and not source.is_file():continue
        if not source.is_file():raise FileNotFoundError(source)
        files[(rgb_relative/name).as_posix()]=source
    if not Path(inception).is_file():raise FileNotFoundError(inception)
    files['runs/D/torch_cache/hub/checkpoints/'+Path(inception).name]=Path(inception)
    names=('rgb_prior_seed42_loss_v1.json','cycle_manifest.json','rgb_manifest.json')
    payloads={}
    originals={name:ROOT/'runs/learned_color'/('data_seed42_v1/'+name if name!='rgb_prior_seed42_loss_v1.json' else name) for name in names}
    if all(p.is_file() for p in originals.values()):
        payloads={f'{ASSET_DIR}/{name}':p.read_bytes() for name,p in originals.items()}
    else:
        with tarfile.open(prior_archive,'r:gz') as archive:
            for name,path in originals.items():
                payload=archive.extractfile(path.relative_to(ROOT).as_posix()).read()
                if path.is_file() and payload!=path.read_bytes():raise ValueError('Frozen artifact differs from current local file: '+str(path))
                payloads[f'{ASSET_DIR}/{name}']=payload
    return files,payloads


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--rgb-snapshot',type=Path,default=ROOT/'runs/pretrained/segformer_cityscapes_b2'/REVISION)
    parser.add_argument('--inception',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
    parser.add_argument('--prior-archive',type=Path,default=ROOT/'runs/checkpoints/rgb_prior_clarity_weights_20261028.tar.gz')
    args=parser.parse_args()
    checksum=args.output.with_name(args.output.name+'.sha256')
    if not args.output.name.endswith('.tar.gz') or args.output.exists() or checksum.exists():parser.error('Preserve existing packages; choose a new .tar.gz')
    files,payloads=collect(args.cache_dir,args.rgb_snapshot,args.inception,args.prior_archive)
    hashes={name:sha256(path) for name,path in files.items()}
    hashes.update({name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()})
    manifest=dict(format='D_ubuntu_offline_assets_v1',files=hashes,
        fmb_included=False,rgb_warmup_included=False,
        note='Reuse the existing server FMB checkpoint. Generate RGB warm-up locally for the new run. Original learned_color artifacts are not overwritten.')
    payloads[f'{ASSET_DIR}/asset_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'x:gz',compresslevel=1) as archive:
        for name,path in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o644
            with path.open('rb') as handle:archive.addfile(info,handle)
            print('[ASSET] '+name,flush=True)
        for name,data in sorted(payloads.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
    checksum.write_text(f'{sha256(args.output)}  {args.output.name}\n',encoding='utf-8',newline='\n')
    print(f'{args.output}: {len(hashes)} assets; {args.output.stat().st_size} bytes; {checksum}',flush=True)


if __name__=='__main__':main()
