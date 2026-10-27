"""Short RGB self-paired adaptation; frozen diffusion, seed42, resumable state."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .common import ROOT, DEFAULT_CACHE, load_config, read_json, write_json, sha256, control_tensor
from .prepare_rgb import DEFAULT_RGB
from .adaptation import configure_trainable, noise_prediction, save_adapter, load_adapter

PILOT_CONFIG = ROOT / "D/configs/sd15_rgb_pilot_seed42.json"


def read_rgb_cache(folder):
    manifest = read_json(folder / "manifest.json")
    if (manifest.get("format") != "thermal_D_rgb_selfpairs_v1" or manifest.get("seed") != 42 or
        read_json(folder / "status.json").get("phase") != "completed"):
        raise ValueError("Prepare a completed seed42 RGB cache first")
    samples, roles = [], {}
    for frame in manifest["frames"]:
        if frame["role"] not in ("train", "val"):
            raise ValueError("Training cache must not contain test images")
        group = frame["group"]
        if group in roles and roles[group] != frame["role"]:
            raise ValueError("Video group leaks between RGB training and validation")
        roles[group] = frame["role"]
        path = folder / frame["condition"]
        if sha256(path) != frame["condition_sha256"]:
            raise ValueError("RGB cache changed")
        with np.load(path, allow_pickle=False) as data:
            rgb, edge = data["RGB_uint8"].copy(), data["E"].copy()
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or edge.shape != rgb.shape[:2]
            or not np.isfinite(edge).all() or np.any((edge < 0) | (edge > 1))):
            raise ValueError("Invalid RGB/soft-E geometry or values")
        samples.append((frame, rgb, edge))
    if {s[0]["role"] for s in samples} != {"train", "val"}:
        raise ValueError("Both training and validation RGB are required")
    return manifest, samples


def encode_cache(pipeline, samples, config, device):
    """Cache posterior parameters, not one fixed noisy latent; no gradients here."""
    result = []
    pipeline.vae.to(device)
    with torch.no_grad():
        for index, (frame, rgb, edge) in enumerate(samples):
            w, h = config["width"], config["height"]
            if rgb.shape[1] * h != rgb.shape[0] * w:
                raise ValueError("Pilot resolution must preserve the source aspect ratio")
            image = Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC)
            values = np.asarray(image).astype(np.float32) / 127.5 - 1
            pixels = torch.from_numpy(values.transpose(2, 0, 1).copy())[None].to(device, dtype=torch.float16)
            posterior = pipeline.vae.encode(pixels).latent_dist
            result.append(dict(mean=posterior.mean.float().cpu(), logvar=posterior.logvar.float().cpu()))
            if (index + 1) % 32 == 0 or index + 1 == len(samples):
                print(f"Latent cache [{index+1}/{len(samples)}]", flush=True)
    scale = pipeline.vae.config.scaling_factor
    pipeline.vae.to("cpu")
    pipeline.text_encoder.to(device)
    with torch.no_grad():
        tokens = pipeline.tokenizer([config["prompt"], ""], padding="max_length",
                      max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text = pipeline.text_encoder(tokens.input_ids.to(device))[0].detach().cpu()
    pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    return dict(posteriors=result, text=text, scaling_factor=scale)


def save_checkpoint(folder, step, controlnet, optimizer, scaler, noise_rng, sampler_rng, order, position, identity):
    final = folder / "checkpoints" / f"step_{step:06d}"
    final.parent.mkdir(exist_ok=True)
    if final.exists():
        raise ValueError("Preserve existing checkpoint")
    pending = Path(tempfile.mkdtemp(prefix=f".step_{step:06d}_", dir=final.parent))
    save_adapter(controlnet, pending / "adapter.safetensors")
    torch.save(dict(step=step, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    noise_rng=noise_rng.get_state(), sampler_rng=sampler_rng.get_state(),
                    order=order, position=position, cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state()), pending / "state.pt")
    write_json(pending / "checkpoint.json", dict(format="D_partial_controlnet_v1", step=step,
              run_sha256=sha256(folder / "run.json"), adapter_sha256=sha256(pending / "adapter.safetensors"),
              state_sha256=sha256(pending / "state.pt"), model=identity["model"], training="RGB self-pairs only"))
    pending.rename(final)
    write_json(folder / "latest.json", dict(step=step, checkpoint=final.relative_to(folder).as_posix()))
    print(f"Saved {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="D_rgb_softE_pilot_seed42")
    parser.add_argument("--rgb-cache", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--config", type=Path, default=PILOT_CONFIG)
    parser.add_argument("--steps", type=int, default=1000, help="Optimizer updates, not epochs")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or args.name in (".", "..") or
        min(args.steps, args.save_every, args.accumulation) < 1 or not 0 < args.lr < 1):
        parser.error("Invalid name, steps or learning rate")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config = load_config(args.config)
    if config["seed"] != 42:
        parser.error("This pilot uses seed42")
    manifest, samples = read_rgb_cache(args.rgb_cache)
    identity = dict(format="D_rgb_adaptation_run_v1", model=config, seed=42,
                    rgb_manifest_sha256=sha256(args.rgb_cache / "manifest.json"),
                    training=dict(steps=args.steps, save_every=args.save_every, lr=args.lr, accumulation=args.accumulation,
                                  batch_size=1, max_grad_norm=1., prompt_dropout=.1,
                                  trainable="ControlNet hint encoder + down/mid residual projections", warmup_steps=50),
                    torch=torch.__version__, code_sha256={p.name:sha256(p) for p in
                       [Path(__file__), ROOT / "D/adaptation.py", ROOT / "D/prepare_rgb.py"]},
                    rgb_source=manifest, limits="RGB self-paired warm-up, not completed TIR-domain adaptation")
    folder = ROOT / "runs/D/training" / args.name
    if args.resume:
        if read_json(folder / "run.json") != identity:
            parser.error("Resume must preserve model, data, schedule and source")
    else:
        if folder.exists():
            parser.error("Preserve existing run; use --resume or a new name")
        folder.mkdir(parents=True)
        write_json(folder / "run.json", identity)
    from diffusers import DDPMScheduler
    from .models import load_pipeline
    pipeline = load_pipeline(config, args.cache_dir, offline=True)
    latent_path = folder / "latents.pt"
    if latent_path.exists():
        cache = torch.load(latent_path, map_location="cpu", weights_only=True)
    else:
        cache = encode_cache(pipeline, samples, config, "cuda:0")
        torch.save(cache, latent_path)
    controlnet, unet = pipeline.controlnet, pipeline.unet
    parameters = configure_trainable(controlnet, unet)
    controlnet.to("cuda:0"); unet.to("cuda:0")
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    noise_rng = torch.Generator("cpu").manual_seed(42)
    sampler_rng = torch.Generator("cpu").manual_seed(42)
    training = [i for i, sample in enumerate(samples) if sample[0]["role"] == "train"]
    order, position, step = [], 0, 0
    if args.resume:
        checkpoint = folder / read_json(folder / "latest.json")["checkpoint"]
        header = read_json(checkpoint / "checkpoint.json")
        if (header["run_sha256"] != sha256(folder / "run.json") or
            header["adapter_sha256"] != sha256(checkpoint / "adapter.safetensors") or
            header["state_sha256"] != sha256(checkpoint / "state.pt")):
            raise ValueError("Checkpoint metadata/weights changed")
        load_adapter(controlnet, checkpoint / "adapter.safetensors")
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        noise_rng.set_state(state["noise_rng"]); sampler_rng.set_state(state["sampler_rng"])
        order, position, step = state["order"], state["position"], state["step"]
        torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
    print(f"Trainable {sum(p.numel() for p in parameters):,} parameters; RGB train={len(training)}; "
          f"update {step}/{args.steps}; resolution={config['width']}x{config['height']}", flush=True)
    text = cache["text"].to("cuda:0", dtype=torch.float16)
    optimizer.zero_grad(set_to_none=True)
    started, logged_loss = time.perf_counter(), 0.
    while step < args.steps:
        total_loss = 0.
        for _ in range(args.accumulation):
            if position >= len(order):
                order = torch.randperm(len(training), generator=sampler_rng).tolist(); position = 0
            index = training[order[position]]; position += 1
            posterior = cache["posteriors"][index]
            latent = (posterior["mean"] + (.5 * posterior["logvar"]).exp() *
                      torch.randn(posterior["mean"].shape, generator=noise_rng)) * cache["scaling_factor"]
            noise = torch.randn(latent.shape, generator=noise_rng)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=noise_rng)
            empty = int(torch.rand((), generator=noise_rng) < .1)
            latent, noise, t = latent.to("cuda:0", dtype=torch.float16), noise.to("cuda:0", dtype=torch.float16), t.to("cuda:0")
            condition = control_tensor(samples[index][2], (config["width"], config["height"])).to("cuda:0", dtype=torch.float16)
            target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latent, noise, t)
            if scheduler.config.prediction_type not in ("epsilon", "v_prediction"):
                raise ValueError("Unsupported scheduler prediction type")
            with torch.autocast("cuda", dtype=torch.float16):
                predicted = noise_prediction(controlnet, unet, scheduler.add_noise(latent, noise, t), t,
                                             text[empty:empty+1], condition)
                loss = F.mse_loss(predicted.float(), target.float()) / args.accumulation
            scaler.scale(loss).backward()
            total_loss += loss.detach()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite adapter gradients; checkpoint preserved, stop for diagnosis")
        old_scale = scaler.get_scale()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min((step + 1) / 50, 1.)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() < old_scale:
            continue
        step += 1
        logged_loss += total_loss
        if step % 10 == 0 or step == args.steps:
            row = dict(step=step, noise_mse=float(total_loss), elapsed=time.perf_counter()-started,
                       lr=optimizer.param_groups[0]["lr"], max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            with (folder / "loss.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(__import__("json").dumps(row) + "\n")
            print(row, flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(folder, step, controlnet, optimizer, scaler, noise_rng, sampler_rng, order, position, identity)
    write_json(folder / "status.json", dict(phase="completed", steps=step))
    print("Next: python -m D.review --run-dir", folder, flush=True)


if __name__ == "__main__":
    main()
