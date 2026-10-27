"""Pinned pretrained components. D0/D1 share one ControlNet img2img path."""

from pathlib import Path

from .common import DEFAULT_CACHE


def load_pipeline(config, cache=DEFAULT_CACHE, *, offline=True):
    import torch
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline
    shared = dict(cache_dir=str(cache), local_files_only=offline, torch_dtype=torch.float16,
                  variant="fp16", use_safetensors=True)
    controlnet = ControlNetModel.from_pretrained(config["control_model"],
                   revision=config["control_revision"], **shared)
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(config["base_model"],
                   revision=config["base_revision"], controlnet=controlnet, **shared)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    for name in ("vae", "unet", "text_encoder", "controlnet", "safety_checker"):
        model = getattr(pipeline, name, None)
        if model is not None:
            model.eval().requires_grad_(False)
    return pipeline


def configure_device(pipeline, device="cuda:0", offload=True):
    import torch
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The full D preview requires CUDA; CPU is supported for preparation/tests")
    # Avoid tiled VAE decoding: its tile boundaries could confound the seam review.
    pipeline.disable_vae_tiling()
    if offload:
        pipeline.enable_model_cpu_offload(gpu_id=target.index or 0)
    else:
        pipeline.to(target)
    return pipeline


def download(config, cache=DEFAULT_CACHE):
    from huggingface_hub import snapshot_download
    base_patterns = ["README.md", "LICENSE*", "model_index.json", "scheduler/scheduler_config.json",
                     "tokenizer/*.json", "tokenizer/*.txt", "feature_extractor/preprocessor_config.json",
                     "*/config.json", "*/*.fp16.safetensors"]
    control_patterns = ["README.md", "LICENSE*", "config.json", "diffusion_pytorch_model.fp16.safetensors"]
    paths = {}
    for kind, patterns in (("base", base_patterns), ("control", control_patterns)):
        paths[kind] = str(Path(snapshot_download(config[f"{kind}_model"],
                         revision=config[f"{kind}_revision"], cache_dir=str(cache),
                         allow_patterns=patterns)).resolve())
    return paths
