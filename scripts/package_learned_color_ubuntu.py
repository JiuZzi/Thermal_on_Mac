#!/usr/bin/env python3
"""Small source-only update; no dataset, learned development bank or weights."""

import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile

from package_semantic_color_ubuntu import EXCLUDED_DIRS,SOURCE_SUFFIXES,EXTRAS

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='learned_color_update_manifest.json'
NEW_FILES=['scripts/learn_color_prior.py','scripts/prepare_grouped_color_data.py','scripts/run_learned_color.py',
           'scripts/check_learned_color.py','scripts/package_learned_color_ubuntu.py',
           'scripts/requirements_learned_color_teacher.txt','scripts/tests/test_color_prior_learning.py',
           'scripts/tests/test_learned_color_launcher.py',
           'docs/RGB学得颜色先验实现与训练.md','docs/微量颜色损失Ubuntu训练.md',
           'docs/从零训练学得颜色先验Ubuntu完整流程.md',
           'configs/learned_color/regularized_v1.json']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--verify-installed',action='store_true')
    args=parser.parse_args()
    if args.verify_installed:
        manifest=json.loads((ROOT/MANIFEST).read_text())
        if manifest['format']!='learned_color_source_update_v1': raise ValueError('Wrong update manifest')
        for name,expected in manifest['files'].items():
            part=PurePosixPath(name)
            if part.is_absolute() or '..' in part.parts or '\\' in name: raise ValueError('Unsafe source path')
            path=ROOT/name
            if not path.is_file() or sha(path)!=expected: raise ValueError(f'Missing/changed source: {name}')
        print(f"Verified {len(manifest['files'])} source files; data, teacher and GAN weights need separate checks.")
        return
    if not args.output or not args.output.name.endswith('.tar.gz') or args.output.exists():
        parser.error('Choose a new --output ending in .tar.gz')
    checksum=args.output.with_name(args.output.name+'.sha256')
    if checksum.exists(): parser.error('Preserve existing checksum')
    paths={ROOT/name for name in (*EXTRAS,*NEW_FILES)}
    for folder in ('pytorch-CycleGAN-and-pix2pix','scripts/semantic_preview','scripts/color_prior_learning'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and not EXCLUDED_DIRS.intersection(path.relative_to(ROOT).parts) and (path.suffix in SOURCE_SUFFIXES or path.name in ('LICENSE','NOTICE')):
                paths.add(path)
    if any(not path.is_file() or path.is_symlink() for path in paths): raise ValueError('Missing/unsafe source file')
    manifest=dict(format='learned_color_source_update_v1',
                  scope='source/config/docs only; no development prior or external weights',files={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'x:gz') as archive:
        for path in sorted(paths):
            name=path.relative_to(ROOT).as_posix(); data=path.read_bytes()
            info=tarfile.TarInfo(name); info.size=len(data); info.mode=0o644
            archive.addfile(info,io.BytesIO(data)); manifest['files'][name]=hashlib.sha256(data).hexdigest()
        payload=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        info=tarfile.TarInfo(MANIFEST); info.size=len(payload); info.mode=0o644
        archive.addfile(info,io.BytesIO(payload))
    checksum.write_text(f'{sha(args.output)}  {args.output.name}\n')
    print(f'{args.output}: {len(paths)} source files; {args.output.stat().st_size} bytes; checksum {checksum}')


if __name__=='__main__': main()
