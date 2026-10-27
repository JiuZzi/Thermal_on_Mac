"""D0: control scale 0; D1: unchanged clarity E. No training or color bank."""

import argparse
from importlib import metadata
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from .common import (ROOT, DEFAULT_CONFIG, DEFAULT_CONDITIONS, DEFAULT_OUTPUT, DEFAULT_CACHE,
                     control_tensor, load_config, read_condition, read_json, sha256, write_json)


def generate_one(pipeline, initial, edge, config, variant):
    """Reset a CPU generator to the SAME seed for every variant/source pair."""
    import torch
    if variant not in ("D0", "D1"):
        raise ValueError("Unknown D variant")
    size = (config["width"], config["height"])
    if initial.width * size[1] != initial.height * size[0]:
        raise ValueError("Model dimensions must preserve the TIR aspect ratio; no stretching or cropping")
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    with torch.inference_mode():
        result = pipeline(prompt=config["prompt"], negative_prompt=config["negative_prompt"],
                          image=initial.resize(size, Image.Resampling.BICUBIC),
                          control_image=control_tensor(edge, size),
                          width=size[0], height=size[1],
                          num_inference_steps=config["steps"], strength=config["strength"],
                          guidance_scale=config["guidance_scale"], eta=config["eta"],
                          controlnet_conditioning_scale=0.0 if variant == "D0" else float(config["control_scale"]),
                          control_guidance_start=0.0, control_guidance_end=1.0, guess_mode=False,
                          generator=generator, num_images_per_prompt=1, output_type="pil")
    if len(result.images) != 1 or result.images[0].size != size:
        raise ValueError("Unexpected diffusion output count/geometry")
    image = result.images[0].convert("RGB")
    flagged = bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected is not None else None
    return image, flagged


def initialize_output(output, identity, resume):
    output = Path(output)
    if output.exists():
        if not resume:
            raise ValueError("Preserve existing results; use --resume or a new output-dir")
        if read_json(output / "run.json") != identity:
            raise ValueError("Resume requires identical conditions, config, code, runtime and variants")
    else:
        if resume:
            raise ValueError("Nothing to resume")
        output.mkdir(parents=True)
        write_json(output / "run.json", identity)
        write_json(output / "records.json", {})
    for variant in identity["variants"]:
        (output / variant / "images").mkdir(parents=True, exist_ok=True)
        (output / variant / "model_resolution").mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--variants", choices=("both", "D0", "D1"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without loading/downloading diffusion weights")
    parser.add_argument("--download-models", action="store_true", help="Download pinned fp16 weights only, without inference")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.download_models:
        from .models import download
        print(download(config, args.cache_dir))
        return
    source = read_json(args.conditions / "manifest.json")
    status = read_json(args.conditions / "status.json")
    if source.get("format") != "thermal_D_conditions_v1" or status.get("phase") != "completed" or not source["frames"]:
        parser.error("Prepare a completed D condition cache first")
    if source["seed"] != config["seed"]:
        parser.error("Condition selection seed and preview seed must match")
    for frame in source["frames"]:
        read_condition(args.conditions, frame)
        w, h = frame["native_size"]
        if w * config["height"] != h * config["width"]:
            parser.error("Model size must preserve the source aspect ratio")
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    if args.check_only:
        print(dict(frames=len(source["frames"]), phase=source["phase"], config=config, packages=versions,
                   message="Inputs valid; pretrained weights have NOT been loaded or downloaded"))
        return
    if "missing" in versions.values():
        parser.error("Install D/requirements.txt in a separate environment")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for full pretrained inference")
    variants = ["D0", "D1"] if args.variants == "both" else [args.variants]
    identity = dict(format="thermal_D_preview_run_v1", config=config, variants=variants,
                    conditions_sha256=sha256(args.conditions / "manifest.json"), source=source,
                    runtime=dict(versions=versions, python=platform.python_version(), system=platform.system(),
                                 gpu=torch.cuda.get_device_name(torch.device(args.device)), dtype="float16",
                                 cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                 model_cpu_offload=not args.no_offload, vae_tiling=False),
                    code_sha256={p.name: sha256(p) for p in sorted((ROOT / "D").glob("*.py"))},
                    limits="Zero-shot pilot; RGB-pretrained Canny control receives soft E; no TIR-domain training; "
                           "D0/D1 measures adding pretrained control, not superiority to ordinary Canny")
    initialize_output(args.output_dir, identity, args.resume)
    records = read_json(args.output_dir / "records.json")
    from .models import load_pipeline, configure_device
    pipeline = configure_device(load_pipeline(config, args.cache_dir, offline=True), args.device, not args.no_offload)
    pipeline.set_progress_bar_config(disable=True)
    started = time.perf_counter()
    total = len(source["frames"]) * len(variants)
    for index, frame in enumerate(source["frames"]):
        initial, edge = read_condition(args.conditions, frame)
        for offset, variant in enumerate(variants, 1):
            key = f"{variant}/{frame['stem']}"
            target = args.output_dir / variant / "images" / f"{frame['stem']}_fake_B.png"
            model_target = args.output_dir / variant / "model_resolution" / f"{frame['stem']}.png"
            if key in records:
                if (not target.is_file() or sha256(target) != records[key]["sha256"] or
                    not model_target.is_file() or sha256(model_target) != records[key]["model_sha256"]):
                    raise ValueError(f"Saved output changed: {key}")
                print(f"Skip verified {key}", flush=True)
                continue
            tick = time.perf_counter()
            image, flagged = generate_one(pipeline, initial, edge, config, variant)
            image.save(model_target)
            image.resize(initial.size, Image.Resampling.BICUBIC).save(target)
            records[key] = dict(sha256=sha256(target), model_sha256=sha256(model_target),
                                safety_flagged=flagged, seed=config["seed"], seconds=time.perf_counter() - tick)
            write_json(args.output_dir / "records.json", records)
            write_json(args.output_dir / "status.json", dict(phase="running", completed=len(records), total=total))
            print(f"[{index * len(variants) + offset}/{total}] {variant} {frame['filename']} "
                  f"{records[key]['seconds']:.1f}s safety_flagged={flagged}", flush=True)
    write_json(args.output_dir / "status.json", dict(phase="completed", completed=len(records), total=total,
                                                    seconds_this_invocation=time.perf_counter() - started))
    from .report import render
    render(args.output_dir, args.conditions)
    print(args.output_dir / "index.html", flush=True)


if __name__ == "__main__":
    main()
