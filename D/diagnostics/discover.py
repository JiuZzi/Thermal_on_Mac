"""Read-only inventory of trained D runs and their matching validation caches.

Discovery checks metadata and file presence. The comparison preflight separately
hashes the complete checkpoint, prior, inference code and selected conditions.
"""

import argparse
import json
from pathlib import Path

from D.common import read_json, sha256
from D.unpaired.protocol import DEFAULT_RUNS, DEFAULT_DATA, DEFAULT_PRIOR


def discover(run_root=DEFAULT_RUNS, conditions_root=DEFAULT_DATA.parent, prior=DEFAULT_PRIOR, *, run_dir=None):
    caches = []
    for path in sorted(Path(conditions_root).rglob('manifest.json')):
        try:
            data, status = read_json(path), read_json(path.parent/'status.json')
            frames = data.get('frames', [])
            count = sum(r.get('domain')=='TIR' and r.get('role')=='val' for r in frames)
            if (data.get('format') != 'D_unpaired_conditions_v1' or not count or
                status.get('phase') != 'completed' or
                any(status[k] != len(frames) for k in ('completed','total') if k in status)):
                continue
            caches.append(dict(path=str(path.parent.resolve()), identity=data['identity'],
                manifest_sha256=sha256(path), val_tir=count))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    paths = [Path(run_dir)/'run.json'] if run_dir is not None else sorted(Path(run_root).rglob('run.json'))
    digest = sha256(prior) if Path(prior).is_file() else None
    rows = []
    for path in paths:
        try:
            run = read_json(path)
        except (OSError, ValueError) as error:
            if run_dir is not None: raise ValueError('Cannot read requested run: '+str(path)) from error
            continue
        if run.get('format') != 'D_unpaired_run_v1':
            if run_dir is not None: raise ValueError('Requested run is not a trained formal D run')
            continue
        source = run.get('data',{}).get('source')
        matching = [c for c in caches if c['identity']==source]
        registered = run.get('validation',{}).get('cache_manifest_sha256')
        matching.sort(key=lambda c:(c['manifest_sha256']!=registered,-c['val_tir'],c['path']))
        row = dict(run_dir=str(path.parent.resolve()), name=path.parent.name, latest_step=None,
            prior_matches=digest is not None and digest==run.get('prior_sha256'),
            conditions=[{k:v for k,v in c.items() if k!='identity'} for c in matching],
            best_file_present=(path.parent/'best.json').is_file(), checkpoint_error=None)
        try:
            from D.unpaired.checkpoint import locate
            checkpoint, step = locate(path.parent)
            header = read_json(checkpoint/'checkpoint.json')
            expected = {'generator.safetensors','reverse.safetensors','critics.safetensors','training.pt'}
            if (header.get('format')!='D_unpaired_checkpoint_v1' or header.get('step')!=step or
                header.get('run_sha256')!=sha256(path) or set(header.get('files',{}))!=expected or
                not all((checkpoint/name).is_file() for name in expected)):
                raise ValueError('Incomplete latest checkpoint or mismatching metadata')
            row['latest_step'] = step
        except (OSError, ValueError, KeyError, TypeError) as error:
            row['checkpoint_error'] = str(error)
        rows.append(row)
    return dict(format='edge_use_diagnostic_discovery_v1', runs=rows,
        note='Read-only discovery; latest is not best. Comparison preflight performs SHA checks before loading a model.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_RUNS)
    parser.add_argument('--conditions-root',type=Path,default=DEFAULT_DATA.parent)
    parser.add_argument('--prior',type=Path,default=DEFAULT_PRIOR)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(discover(args.run_root,args.conditions_root,args.prior,run_dir=args.run_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
