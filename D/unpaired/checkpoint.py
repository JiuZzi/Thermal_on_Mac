"""Atomic full training recovery; only learned tensors, not frozen SD weights."""

from pathlib import Path
import tempfile
import random
import re
import shutil
import time
import warnings

import numpy as np

import torch
from safetensors.torch import save_file, load_file

from D.common import sha256, read_json, write_json


def capture_rng():
    nr = np.random.get_state()
    return dict(cpu_rng=torch.get_rng_state(),
                # CPU-only checks must not initialize a CUDA context just
                # because this machine also has an available NVIDIA GPU.
                cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
                python_rng=random.getstate(),
                numpy_rng=dict(name=nr[0], keys=torch.tensor(nr[1].astype(np.int64)), position=nr[2],
                               has_gauss=nr[3], cached_gauss=nr[4]))


def restore_rng(state):
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    nr = state["numpy_rng"]
    np.random.set_state((nr["name"], nr["keys"].numpy().astype(np.uint32), nr["position"], nr["has_gauss"], nr["cached_gauss"]))


def _finite_state(values):
    result = {k:v.detach().cpu().contiguous() for k,v in values.items()}
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError("Refuse to checkpoint nonfinite model weights")
    return result


def _prune_recovery(folder, keep=2):
    # This is the ONLY deletion performed here. Never prune archival epochs.
    root = folder / "recovery"
    if root.is_symlink() or root.resolve().parent != folder.resolve():
        raise ValueError("Unsafe recovery directory")
    candidates = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"step_\d+_\d+", p.name)]
    latest = (folder / read_json(folder / "latest.json")["checkpoint"]).resolve()
    # Protect the committed pointer even with coarse timestamps or clock changes.
    candidates.sort(key=lambda p:(p.resolve()==latest, *map(int,p.name.split("_")[1:])), reverse=True)
    for path in candidates[keep:]:
        if path.is_symlink() or path.resolve().parent != root.resolve():
            raise ValueError("Refuse to prune outside this run's recovery directory")
        shutil.rmtree(path.resolve())


def save(folder, step, engine, streams, *, kind="archive", metadata=None):
    directories = dict(archive="checkpoints", recovery="recovery", emergency="emergency")
    if kind not in directories:
        raise ValueError("Unknown checkpoint kind")
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint update count must be a nonnegative integer")
    if getattr(engine, "phase", "ready") == "commit":
        raise ValueError("Refuse checkpoint during an uncertain optimizer commit")
    engine_state = engine.state()
    engine.validate_state(engine_state)
    name = f"step_{step:06d}" + (f"_{time.time_ns()}" if kind != "archive" else "")
    target = folder / directories[kind] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Preserve existing checkpoint")
    pending = Path(tempfile.mkdtemp(prefix=f".step_{step:06d}_", dir=target.parent))
    try:
        save_file(_finite_state(engine.generator.export_state()), str(pending / "generator.safetensors"))
        save_file(_finite_state(engine.reverse.state_dict()), str(pending / "reverse.safetensors"))
        save_file(_finite_state(engine.critics.state_dict()), str(pending / "critics.safetensors"))
        torch.save(dict(step=step, engine=engine_state, streams={k:v.state() for k,v in streams.items()},
                        **capture_rng()), pending / "training.pt")
        files = {p.name:sha256(p) for p in pending.iterdir()}
        write_json(pending / "checkpoint.json", dict(format="D_unpaired_checkpoint_v1", step=step,
                   run_sha256=sha256(folder / "run.json"), files=files, kind=kind, metadata=metadata))
        pending.rename(target)
        write_json(folder / "latest.json", dict(step=step, checkpoint=target.relative_to(folder).as_posix()))
    except BaseException:
        # Only this call's uncommitted temporary directory is disposable. Keep
        # complete snapshots and the previous latest pointer intact on failure.
        if pending.exists():
            try:
                shutil.rmtree(pending)
            except OSError:
                pass
        raise
    if kind == "recovery":
        try:
            _prune_recovery(folder)
        except OSError as error:
            warnings.warn(f"Recovery saved; old snapshot cleanup deferred: {error}")
    return target


def locate(folder, step=None):
    if step is None:
        latest = read_json(folder / "latest.json")
        path = folder / latest["checkpoint"]
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Checkpoint pointer leaves this run")
        if read_json(path / "checkpoint.json")["step"] != latest["step"]:
            raise ValueError("Checkpoint pointer update count mismatch")
        return path, latest["step"]
    archive = folder / "checkpoints" / f"step_{step:06d}"
    if archive.is_dir():
        return archive, step
    candidates = [p for kind in ("recovery", "emergency")
                  for p in (folder / kind).glob(f"step_{step:06d}_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint for update {step}")
    return max(candidates, key=lambda p:p.stat().st_mtime_ns), step


def verify(folder, checkpoint):
    header = read_json(checkpoint / "checkpoint.json")
    if header.get("format") != "D_unpaired_checkpoint_v1" or header["run_sha256"] != sha256(folder / "run.json"):
        raise ValueError("Checkpoint/run mismatch")
    if set(header["files"]) != {"generator.safetensors", "reverse.safetensors", "critics.safetensors", "training.pt"}:
        raise ValueError("Incomplete checkpoint file manifest")
    for name, value in header["files"].items():
        if Path(name).name != name or sha256(checkpoint / name) != value:
            raise ValueError(f"Checkpoint file changed: {name}")
    return header


def restore(folder, checkpoint, engine, streams):
    header = verify(folder, checkpoint)
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
    if state["step"] != header["step"]:
        raise ValueError("Checkpoint update counts differ")
    engine.validate_state(state["engine"])
    # Validate all learned tensors before mutating the live networks. A legacy
    # checkpoint can have valid file hashes yet contain nonfinite tensors.
    learned = {name: _finite_state(load_file(str(checkpoint / f"{name}.safetensors"), device="cpu"))
               for name in ("generator", "reverse", "critics")}
    for name, expected in (("generator", engine.generator.export_state()),
                           ("reverse", engine.reverse.state_dict()), ("critics", engine.critics.state_dict())):
        if set(learned[name]) != set(expected) or any(learned[name][k].shape != v.shape for k, v in expected.items()):
            raise ValueError(f"Checkpoint tensor keys/shapes changed: {name}")
    engine.generator.import_state(learned["generator"])
    engine.reverse.load_state_dict(learned["reverse"], strict=True)
    engine.critics.load_state_dict(learned["critics"], strict=True)
    engine.restore(state["engine"])
    for key, stream in streams.items():
        stream.restore(state["streams"][key])
    restore_rng(state)
    return state["step"]
