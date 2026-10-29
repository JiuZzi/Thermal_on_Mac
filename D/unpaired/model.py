"""Differentiable DDIM translator with [TIR,E,0], semantics and color tokens."""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from D.adaptation import configure_trainable, load_adapter
from D.common import sha256, read_json
from .semantics import legacy_models


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, values):
        output = self.base(values)
        return output + self.up(self.down(values.float())).to(output.dtype)


def install_lora(unet, rank):
    if not rank:
        return
    for name, module in list(unet.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith("attn2." + suffix)
                for suffix in ("to_q", "to_k", "to_v", "to_out.0")):
            parent, child = name.rsplit(".", 1)
            setattr(unet.get_submodule(parent), child, LoRALinear(module, rank))


def convert_edge_encoder(controlnet):
    """Preserve old [E,E,E] response; learn TIR channel from zero residual."""
    conv = controlnet.controlnet_cond_embedding.conv_in
    if conv.in_channels != 3:
        raise ValueError("Expected three-channel ControlNet condition encoder")
    with torch.no_grad():
        edge_weight = conv.weight.float().sum(1, keepdim=True)
        conv.weight.zero_()
        conv.weight[:, 1:2].copy_(edge_weight.to(conv.weight))


class SourceAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
                                    nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
                                    nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(), nn.Conv2d(64, 4, 3, 1, 1))
        nn.init.zeros_(self.layers[-1].weight); nn.init.zeros_(self.layers[-1].bias)
        with torch.no_grad():
            self.layers[0].weight[:, 2].zero_()

    def forward(self, condition):
        return self.layers(condition.float())


class SemanticColorFusion(nn.Module):
    def __init__(self, controlnet, prior, width=16, *, policy="legacy", min_quality=.1,
                 semantic_gate=True):
        super().__init__()
        if policy not in ("legacy", "quality_masked") or not 0 <= min_quality <= 1:
            raise ValueError("Invalid semantic/color fusion policy")
        self.prior = prior
        self.policy, self.min_quality, self.semantic_gate = policy, min_quality, semantic_gate
        self.collect_diagnostics = False
        self.diagnostics = {}
        def encoder(channels):
            return nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.semantic = encoder(15)
        self.color = encoder(prior.feature_channels)
        channels = [layer.out_channels for layer in controlnet.controlnet_down_blocks]
        channels.append(controlnet.controlnet_mid_block.out_channels)
        self.sem_heads = nn.ModuleList([nn.Conv2d(width, channel, 1) for channel in channels])
        self.color_heads = nn.ModuleList([nn.Conv2d(width, channel, 1) for channel in channels])
        for layer in [*self.sem_heads, *self.color_heads]:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def encode(self, p, q, size):
        color = self.prior(p, q)
        if self.policy == "quality_masked":
            # Pool only supported observations. Normalizing by support avoids
            # squaring Q: the output gate below applies the strength once.
            valid = (q > 0) & (q >= self.min_quality)
            quality = q.detach().float() * valid
            semantic = self._masked_encode(self.semantic, p.float(), quality, size) \
                if self.semantic_gate else self.semantic(F.interpolate(p.float(), size=size, mode="area"))
            gate = color["strength"].float() * valid
            learned = self._masked_encode(self.color, color["features"].float(), gate, size)
            return semantic, learned, quality, gate
        semantic = self.semantic(F.interpolate(p.float(), size=size, mode="area"))
        learned = self.color(F.interpolate(color["features"].float(), size=size, mode="area"))
        return semantic, learned, q, color["strength"]

    @staticmethod
    def _masked_encode(encoder, values, weights, size):
        mass = F.interpolate(weights, size=size, mode="area")
        values = F.interpolate(values * weights, size=size, mode="area") / mass.clamp_min(1e-8)
        support = (mass > 0).to(values)
        # Mask after every layer as well, so a convolution's bias or context at
        # an unsupported cell cannot be propagated back into trusted cells.
        for layer in encoder:
            values = layer(values) * support
        return values

    def inject(self, down, mid, encoded, config):
        semantic, color, quality, gate = encoded
        outputs = []
        norms = []
        for index, original in enumerate([*down, mid]):
            value = original
            size = value.shape[-2:]
            sem_addition = color_addition = None
            if config.get("use_semantic", True):
                addition = self.sem_heads[index](F.interpolate(semantic, size=size, mode="nearest"))
                if config["semantic_gate"]:
                    addition = addition * F.interpolate(quality, size=size, mode="area")
                sem_addition = addition
                value = value + addition.to(value.dtype)
            if config.get("use_color", True):
                addition = self.color_heads[index](F.interpolate(color, size=size, mode="nearest"))
                color_addition = addition * F.interpolate(gate, size=size, mode="area")
                value = value + color_addition.to(value.dtype)
            outputs.append(value)
            if self.collect_diagnostics:
                def rms(tensor):
                    return tensor.detach().float().square().mean().sqrt() if tensor is not None else value.new_zeros(())
                norms.append(torch.stack((rms(original), rms(sem_addition), rms(color_addition))))
        if self.collect_diagnostics:
            control, sem, chroma = torch.stack(norms).mean(0).unbind()
            self.diagnostics = dict(control_residual_rms=control, semantic_residual_rms=sem,
                                    color_residual_rms=chroma,
                                    semantic_control_ratio=sem / control.clamp_min(1e-8),
                                    color_control_ratio=chroma / control.clamp_min(1e-8))
        return outputs[:-1], outputs[-1]


class FormalGenerator(nn.Module):
    def __init__(self, pipeline, config, prior_path, source_identity, *, rgb_init=None):
        super().__init__()
        legacy_models()
        from models.learned_color_prior import LearnedChromaPrior
        from models.semantic_edges import FMB_CLASS_NAMES
        self.unet, self.vae, self.controlnet = pipeline.unet, pipeline.vae, pipeline.controlnet
        self.config = config
        self.rgb_init_info = None
        if rgb_init is not None:
            header = read_json(rgb_init / "checkpoint.json")
            if header["format"] != "D_partial_controlnet_v1" or sha256(rgb_init / "adapter.safetensors") != header["adapter_sha256"]:
                raise ValueError("Invalid RGB initialization checkpoint")
            for key in ("base_model", "base_revision", "control_model", "control_revision"):
                if header["model"][key] != config[key]:
                    raise ValueError("RGB initialization uses different pretrained models")
            load_adapter(self.controlnet, rgb_init / "adapter.safetensors")
            self.rgb_init_info = dict(step=header["step"], sha256=header["adapter_sha256"])
        convert_edge_encoder(self.controlnet)
        configure_trainable(self.controlnet, self.unet)
        install_lora(self.unet, config["lora_rank"])
        self.vae.requires_grad_(False)
        self.vae.enable_gradient_checkpointing()
        self.source = SourceAdapter()
        prior = LearnedChromaPrior(prior_path, FMB_CLASS_NAMES,
                                  require_loss_support=config["loss_weights"]["color_prior"] > 0)
        self.fusion = SemanticColorFusion(self.controlnet, prior, config["semantic_width"],
            policy=config.get("fusion_policy", "legacy"), min_quality=config.get("fusion_min_quality", .1),
            semantic_gate=config["semantic_gate"])
        self.output_diagnostics = {}
        pipeline.text_encoder.float().cpu().eval()
        tokens = pipeline.tokenizer([config["negative_prompt"], config["prompt"]], padding="max_length",
                     max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text = pipeline.text_encoder(tokens.input_ids)[0].detach().clone()
        self.register_buffer("text", text, persistent=False)
        from diffusers import DDIMScheduler, DDPMScheduler
        self.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        self.noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        self.identity = dict(interface="TIR01,clarity_E01,reserved_zero", confidence=False,
            base=config["base_revision"], control=config["control_revision"],
            lora_rank=config["lora_rank"], semantic_width=config["semantic_width"],
            prior=prior.metadata, source=source_identity, semantic_gate=config["semantic_gate"],
            use_semantic=config.get("use_semantic", True), use_color=config.get("use_color", True))
        if config.get("format") == "D_unpaired_v2":
            self.identity.update(fusion_policy=config["fusion_policy"],
                                 fusion_min_quality=config["fusion_min_quality"],
                                 chroma_smooth_policy=config["chroma_smooth_policy"])
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).digest()
        self.register_buffer("input_signature", torch.tensor(list(digest), dtype=torch.uint8))

    @property
    def prior(self):
        return self.fusion.prior

    def export_state(self):
        selected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu().contiguous().clone() for name, value in self.state_dict().items()
                if name in selected or name == "input_signature"}

    def import_state(self, state):
        expected = self.export_state()
        if set(state) != set(expected) or not torch.equal(state["input_signature"].cpu(), expected["input_signature"].cpu()):
            raise ValueError("Formal D signature/keys mismatch; do not load an E-only/GAN checkpoint here")
        current = self.state_dict()
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != expected[name].shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid formal D weight: {name}")
                current[name].copy_(value.to(current[name]))

    def _validate(self, x, p, q):
        if (x.ndim != 4 or x.shape[0] < 1 or x.shape[1] != 3 or min(x.shape[-2:]) < 32 or
            any(v % 8 for v in x.shape[-2:]) or not torch.isfinite(x).all() or
            (x < 0).any() or (x > 1).any() or torch.count_nonzero(x[:, 2])):
            raise ValueError("Expected [TIR01,E01,zero] in aligned coordinates divisible by 8")
        if p.shape != (x.shape[0], 15, *x.shape[-2:]) or q.shape != x[:, :1].shape:
            raise ValueError("Semantic/quality coordinates differ from X")
        for name, values in (("semantic probabilities", p), ("semantic quality", q)):
            if not torch.is_floating_point(values) or not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError(f"Expected finite {name} in [0,1]")
        if not torch.allclose(p.float().sum(1), torch.ones_like(p[:, 0], dtype=torch.float32), atol=4e-6, rtol=0):
            raise ValueError("Semantic probability mass differs from one")

    def _autocast(self, x):
        return torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda")

    def _encode(self, rgb):
        def run(value):
            return self.vae.encode(value).latent_dist.mode() * self.vae.config.scaling_factor
        if torch.is_grad_enabled() and rgb.requires_grad:
            return checkpoint(run, rgb, use_reentrant=False)
        with torch.no_grad():
            return run(rgb)

    def _predict(self, latent, timestep, x, encoded, text):
        down, mid = self.controlnet(latent, timestep, encoder_hidden_states=text,
                                    controlnet_cond=x, return_dict=False)
        down, mid = self.fusion.inject(down, mid, encoded, self.config)
        return self.unet(latent, timestep, encoder_hidden_states=text,
                         down_block_additional_residuals=down, mid_block_additional_residual=mid).sample

    def forward(self, x, p, q, *, generator=None):
        self._validate(x, p, q)
        if generator is None:
            generator = torch.Generator("cpu").manual_seed(42)
        with self._autocast(x):
            latent = self._encode(x[:, :1].repeat(1, 3, 1, 1) * 2 - 1)
            latent = latent + self.source(x).to(latent.dtype)
            encoded = self.fusion.encode(p, q, latent.shape[-2:])
            self.scheduler.set_timesteps(self.config["sampling_steps"], device=x.device)
            count = int(self.config["sampling_steps"] * self.config["strength"])
            timesteps = self.scheduler.timesteps[-count:]
            noise = torch.randn(latent.shape, generator=generator, device="cpu").to(latent)
            latent = self.scheduler.add_noise(latent, noise, timesteps[0].expand(x.shape[0]))
            text = self.text.to(latent.dtype)
            for timestep in timesteps:
                predicted = self._predict(latent, timestep, x, encoded, text[1:2].expand(x.shape[0], -1, -1))
                if self.config["guidance_scale"] > 1:
                    unconditional = self._predict(latent, timestep, x, encoded, text[:1].expand(x.shape[0], -1, -1))
                    predicted = unconditional + self.config["guidance_scale"] * (predicted - unconditional)
                latent = self.scheduler.step(predicted, timestep, latent, eta=0., return_dict=False)[0]
            result = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        if result.shape != (x.shape[0], 3, *x.shape[-2:]):
            raise ValueError("Generated RGB geometry changed")
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite decoded RGB before output clamping; inspect model/precision")
        if self.fusion.collect_diagnostics:
            self.output_diagnostics = dict(decoder_saturated_fraction=(result.detach().abs() > 1).float().mean(),
                                           decoder_abs_max=result.detach().float().abs().max())
        return result.float().clamp(-1, 1)

    def replay_loss(self, rgb, x, p, q, generator):
        """RGB self-pair denoising, now with the same three-channel interface."""
        with self._autocast(x):
            with torch.no_grad():
                posterior = self.vae.encode(rgb).latent_dist
                eps = torch.randn(posterior.mean.shape, generator=generator).to(posterior.mean)
                latent = (posterior.mean + posterior.std * eps) * self.vae.config.scaling_factor
            noise = torch.randn(latent.shape, generator=generator).to(latent)
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                              (rgb.shape[0],), generator=generator).to(rgb.device)
            encoded = self.fusion.encode(p, q, latent.shape[-2:])
            predicted = self._predict(self.noise_scheduler.add_noise(latent, noise, t), t, x, encoded,
                                      self.text[1:2].to(latent.dtype).expand(rgb.shape[0], -1, -1))
            kind = self.noise_scheduler.config.prediction_type
            if kind == "epsilon":
                target = noise
            elif kind == "v_prediction":
                target = self.noise_scheduler.get_velocity(latent, noise, t)
            else:
                raise ValueError("Unsupported diffusion prediction type")
            return F.mse_loss(predicted.float(), target.float())
