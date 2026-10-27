"""Atomic full training recovery; only learned tensors, not frozen SD weights."""

from pathlib import Path
import tempfile
import random

import numpy as np

import torch
from safetensors.torch import save_file, load_file

from D.common import sha256, read_json, write_json


def save(folder, step, engine, streams):
    target = folder / "checkpoints" / f"step_{step:06d}"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise ValueError("Preserve existing checkpoint")
    pending = Path(tempfile.mkdtemp(prefix=f".step_{step:06d}_", dir=target.parent))
    save_file(engine.generator.export_state(), str(pending / "generator.safetensors"))
    save_file({k:v.detach().cpu().contiguous() for k,v in engine.reverse.state_dict().items()}, str(pending / "reverse.safetensors"))
    save_file({k:v.detach().cpu().contiguous() for k,v in engine.critics.state_dict().items()}, str(pending / "critics.safetensors"))
    numpy_rng = np.random.get_state()
    torch.save(dict(step=step, engine=engine.state(), streams={k:v.state() for k,v in streams.items()},
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
                    numpy_rng=dict(name=numpy_rng[0], keys=torch.tensor(numpy_rng[1].astype(np.int64)),
                                   position=numpy_rng[2], has_gauss=numpy_rng[3], cached_gauss=numpy_rng[4])), pending / "training.pt")
    files = {p.name:sha256(p) for p in pending.iterdir()}
    write_json(pending / "checkpoint.json", dict(format="D_unpaired_checkpoint_v1", step=step,
               run_sha256=sha256(folder / "run.json"), files=files))
    pending.rename(target)
    write_json(folder / "latest.json", dict(step=step, checkpoint=target.relative_to(folder).as_posix()))
    return target


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
    engine.generator.import_state(load_file(str(checkpoint / "generator.safetensors"), device="cpu"))
    engine.reverse.load_state_dict(load_file(str(checkpoint / "reverse.safetensors")), strict=True)
    engine.critics.load_state_dict(load_file(str(checkpoint / "critics.safetensors")), strict=True)
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
    if state["step"] != header["step"]:
        raise ValueError("Checkpoint update counts differ")
    engine.restore(state["engine"])
    for key, stream in streams.items():
        stream.restore(state["streams"][key])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    nr = state["numpy_rng"]
    np.random.set_state((nr["name"],nr["keys"].numpy().astype(np.uint32),nr["position"],nr["has_gauss"],nr["cached_gauss"]))
    return state["step"]
