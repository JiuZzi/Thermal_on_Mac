"""Sequential-graph unpaired updates, keeping cycle memory bounded."""

import torch
from torch import nn
from torch.nn import functional as F

from .semantics import legacy_models
from .protocol import ReplayPool
from .checkpoint import capture_rng, restore_rng
from .losses import edge_preservation, spatial_content, semantic_consistency, chroma_smoothness, lsgan


class NumericalFailure(RuntimeError):
    """No optimizer has committed this rejected update; full recovery is safe."""
    def __init__(self, report):
        self.report = report
        super().__init__(f"Repeated nonfinite values at {report['stage']}; update was not committed")


class _NonfiniteUpdate(FloatingPointError):
    pass


def auxiliary_networks(device):
    legacy_models()
    from models import networks
    norm = networks.get_norm_layer("instance")
    reverse = networks.ResnetGenerator(3, 1, 64, norm_layer=norm, use_dropout=False, n_blocks=9)
    critics = nn.ModuleDict(dict(RGB=networks.NLayerDiscriminator(3,64,3,norm_layer=norm),
                                 TIR=networks.NLayerDiscriminator(1,64,3,norm_layer=norm)))
    networks.init_weights(reverse, "normal", .02)
    networks.init_weights(critics, "normal", .02)
    return reverse.to(device), critics.to(device)


class UnpairedEngine:
    def __init__(self, generator, reverse, critics, builder, rgb_teacher, config, *, device="cuda:0"):
        self.generator, self.reverse, self.critics = generator, reverse, critics
        self.builder, self.rgb_teacher, self.config = builder, rgb_teacher, config
        self.device = device
        self.params_g = [p for p in generator.parameters() if p.requires_grad] + list(reverse.parameters())
        self.params_d = list(critics.parameters())
        self.opt_g = torch.optim.AdamW([
            dict(params=[p for p in generator.parameters() if p.requires_grad], lr=config["lr_diffusion"], weight_decay=.0001),
            dict(params=reverse.parameters(), lr=config["lr_reverse"], weight_decay=0.)], betas=(.5,.999))
        self.opt_d = torch.optim.Adam(critics.parameters(), lr=config["lr_discriminator"], betas=(.5,.999))
        self.scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"), init_scale=1024.)
        self.noise_rng = torch.Generator("cpu").manual_seed(42)
        self.patch_rng = torch.Generator("cpu").manual_seed(42)
        self.rgb_pool, self.tir_pool = ReplayPool(config["pool_size"]), ReplayPool(config["pool_size"])
        self.overflow_count = 0
        self.numerical_failures_total = 0
        self.phase = "ready"

    def retry_state(self):
        def pool_state(pool):
            state = pool.state()
            return dict(state, images=list(state["images"]))
        return dict(noise_rng=self.noise_rng.get_state(), patch_rng=self.patch_rng.get_state(),
                    rgb_pool=pool_state(self.rgb_pool), tir_pool=pool_state(self.tir_pool))

    def restore_retry(self, state):
        self.noise_rng.set_state(state["noise_rng"]); self.patch_rng.set_state(state["patch_rng"])
        self.rgb_pool.restore(state["rgb_pool"]); self.tir_pool.restore(state["tir_pool"])

    def _require_finite(self, values, stage):
        self.phase = stage
        if not torch.isfinite(values).all():
            raise _NonfiniteUpdate(stage)

    def _check_losses(self, terms, stage, total=None):
        values = [v.detach().float() for v in terms.values()]
        if total is not None:
            values.append(total.detach().float())
        self._require_finite(torch.stack(values), stage)

    def _check_gradients(self, parameters, stage):
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if grads:
            self._require_finite(torch.nn.utils.get_total_norm(grads, foreach=True), stage)

    @staticmethod
    def _gradient_norm(parameters):
        values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        return torch.stack(values).norm() if values else None

    def update(self, tir, rgb, step, total, *, diagnostics=False):
        before = self.retry_state()
        global_before = capture_rng()
        try:
            applied, values = self._attempt_update(tir, rgb, step, total, diagnostics=diagnostics)
        except FloatingPointError as error:
            # An optimizer may have partially committed. Never describe that
            # state as an unchanged, retryable update or checkpoint it as such.
            if self.phase in ("commit", "committed"):
                raise
            bad = []
            for prefix, module in (("generator", self.generator), ("reverse", self.reverse), ("critics", self.critics)):
                for name, parameter in module.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        bad.append(f"{prefix}.{name}")
                        if len(bad) >= 8:
                            break
                if len(bad) >= 8:
                    break
            self.opt_g.zero_grad(set_to_none=True); self.opt_d.zero_grad(set_to_none=True)
            self.restore_retry(before)
            restore_rng(global_before)
            scale_before = self.scaler.get_scale()
            if self.scaler.is_enabled():
                # Initialize lazily even if the failure happened before backward.
                self.scaler.scale(torch.zeros((), device=self.device))
                self.scaler.update(new_scale=max(scale_before * self.scaler.get_backoff_factor(),
                                                 torch.finfo(torch.float32).tiny))
            self.overflow_count += 1; self.numerical_failures_total += 1
            report = dict(amp_overflow=self.overflow_count, stage=self.phase, reason=str(error),
                          amp_scale_before=scale_before, amp_scale_after=self.scaler.get_scale(),
                          nonfinite_gradient_parameters=bad, optimizers_unchanged=True,
                          retry_same_batch=True, numerical_failures_total=self.numerical_failures_total)
            self.phase = "rejected"
            if self.overflow_count > 8:
                raise NumericalFailure(report) from error
            return False, report
        if diagnostics:
            values["amp_scale"] = self.scaler.get_scale()
            values["retry_count"] = self.overflow_count
        self.overflow_count = 0
        # The outer loop acknowledges the counter/sampler transaction before
        # setting ready. An interrupt in between must use a periodic snapshot.
        self.phase = "committed"
        return applied, values

    def _rates(self, step, total):
        warm = min((step+1) / max(self.config["warmup_steps"], 1), 1.)
        decay = min(1., max(0., (total-step) / max(total-total//2, 1)))
        for group, base in zip(self.opt_g.param_groups, (self.config["lr_diffusion"],self.config["lr_reverse"])):
            group["lr"] = base * warm * decay
        for group in self.opt_d.param_groups:
            group["lr"] = self.config["lr_discriminator"] * warm * decay

    def _attempt_update(self, tir, rgb, step, total, *, diagnostics=False):
        g, reverse, critics = self.generator, self.reverse, self.critics
        g.fusion.collect_diagnostics = diagnostics
        weights = self.config["loss_weights"]
        g.train(); reverse.train(); critics.requires_grad_(False)
        self._rates(step, total)
        self.opt_g.zero_grad(set_to_none=True)
        losses = {}
        # First graph: TIR -> RGB -> reconstructed TIR.
        self.phase = "tir_forward"
        fake_rgb = g(tir["x"], tir["p"], tir["q"], generator=self.noise_rng)
        self._require_finite(fake_rgb, "tir_forward")
        if diagnostics:
            losses.update(g.fusion.diagnostics)
            losses.update(g.output_diagnostics)
            losses["rgb_replay_active"] = int(bool(weights["rgb_replay"]) and (step+1) % self.config["replay_every"] == 0)
        real_tir = tir["x"][:, :1] * 2 - 1
        terms = dict(adversarial_rgb=lsgan(critics["RGB"](fake_rgb), True))
        if weights["cycle_tir"]:
            terms["cycle_tir"] = F.l1_loss(reverse(fake_rgb), real_tir)
        if weights["edge"]:
            terms["edge"] = edge_preservation(fake_rgb, tir["x"][:, :1], tir["x"][:, 1:2])
        if weights["content"]:
            terms["content"] = spatial_content(fake_rgb, tir["x"][:, :1], tir["x"][:, 1:2], generator=self.patch_rng)
        if weights["semantic"]:
            predicted = self.rgb_teacher(fake_rgb, side=self.config["semantic_loss_size"])
            terms["semantic"] = semantic_consistency(predicted, tir["p"], tir["q"], self.config["semantic_min_quality"])
        if weights["color_prior"]:
            terms["color_prior"], coverage = g.prior.chroma_loss(fake_rgb, tir["p"], tir["q"],
                                               min_quality=self.config.get("color_min_quality", .5))
            losses["color_coverage"] = coverage.detach()
        if weights["chroma_smooth"]:
            quality = tir["q"] if self.config.get("chroma_smooth_policy", "legacy") == "quality_soft" else None
            terms["chroma_smooth"] = chroma_smoothness(fake_rgb, tir["x"][:,1:2], tir["p"], quality,
                                                      minimum=self.config["semantic_min_quality"])
        color_ramp = min((step+1) / max(1,self.config["color_warmup_steps"]), 1.)
        loss = sum(value * (weights["adversarial"] if key.startswith("adversarial") else weights[key]) *
                   (color_ramp if key == "color_prior" else 1.) for key,value in terms.items())
        self._check_losses(terms, "tir_losses", loss)
        losses["G_total"] = loss.detach()
        if diagnostics:
            losses.update({f"weighted_{key}": (value * (weights["adversarial"] if key.startswith("adversarial") else weights[key]) *
                          (color_ramp if key == "color_prior" else 1.)).detach() for key, value in terms.items()})
            losses["color_weight"] = weights["color_prior"] * color_ramp
        self.scaler.scale(loss).backward()
        self._check_gradients(self.params_g, "tir_backward")
        losses.update({key:value.detach() for key,value in terms.items()})
        saved_rgb = fake_rgb.detach()
        del fake_rgb, loss, terms

        # Second graph: RGB -> synthetic TIR -> reconstructed RGB.
        self.phase = "rgb_reverse_forward"
        fake_tir = reverse(rgb["rgb"])
        self._require_finite(fake_tir, "rgb_reverse_forward")
        terms = dict(adversarial_tir=lsgan(critics["TIR"](fake_tir), True))
        if weights["cycle_rgb"]:
            sx, sp, sq = self.builder.synthetic(fake_tir)
            self.phase = "rgb_cycle_forward"
            reconstructed = g(sx, sp, sq, generator=self.noise_rng)
            self._require_finite(reconstructed, "rgb_cycle_forward")
            terms["cycle_rgb"] = F.l1_loss(reconstructed, rgb["rgb"])
        loss = sum(value * (weights["adversarial"] if key.startswith("adversarial") else weights[key])
                   for key,value in terms.items())
        self._check_losses(terms, "rgb_cycle_losses", loss)
        losses["G_total"] = losses["G_total"] + loss.detach()
        if diagnostics:
            losses.update({f"weighted_{key}": (value * (weights["adversarial"] if key.startswith("adversarial") else weights[key])).detach()
                           for key, value in terms.items()})
        self.scaler.scale(loss).backward()
        self._check_gradients(self.params_g, "rgb_cycle_backward")
        losses.update({key:value.detach() for key,value in terms.items()})
        saved_tir = fake_tir.detach()
        del fake_tir, loss, terms
        if weights["cycle_rgb"]:
            del reconstructed, sx, sp, sq

        # Replay has its own graph; it never uses a randomly paired RGB target.
        if weights["rgb_replay"] and (step+1) % self.config["replay_every"] == 0:
            self.phase = "rgb_replay_forward"
            replay = g.replay_loss(rgb["rgb"], rgb["x"], rgb["p"], rgb["q"], self.noise_rng)
            self._check_losses(dict(replay=replay), "rgb_replay_loss", weights["rgb_replay"] * replay)
            losses["G_total"] = losses["G_total"] + weights["rgb_replay"] * replay.detach()
            if diagnostics:
                losses["weighted_rgb_replay"] = weights["rgb_replay"] * replay.detach()
            self.scaler.scale(weights["rgb_replay"] * replay).backward()
            self._check_gradients(self.params_g, "rgb_replay_backward")
            losses["rgb_replay"] = replay.detach()
            del replay
        self.scaler.unscale_(self.opt_g)
        if diagnostics:
            for name, parameters in (("diffusion", g.parameters()), ("reverse", reverse.parameters())):
                value = self._gradient_norm(parameters)
                if value is not None:
                    losses[f"grad_norm_{name}"] = value
            conv = g.controlnet.controlnet_cond_embedding.conv_in
            if conv.weight.grad is not None:
                for channel,name in enumerate(("TIR", "E", "C_zero")):
                    losses[f"grad_{name}"] = conv.weight.grad[:,channel].norm().detach()
            losses["semantic_quality_coverage"] = (tir["q"] >= self.config["semantic_min_quality"]).float().mean()
            for name, module in (("semantic", g.fusion.sem_heads), ("color", g.fusion.color_heads), ("content", g.source)):
                values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
                losses[f"grad_{name}_adapter"] = torch.stack(values).sum().sqrt() if values else tir["x"].sum()*0
        norm = nn.utils.clip_grad_norm_(self.params_g, self.config["max_grad_norm"])
        self._require_finite(norm, "generator_gradients")

        # Independent unpaired domain critics with original-style replay pools.
        critics.requires_grad_(True)
        self.opt_d.zero_grad(set_to_none=True)
        d_rgb = .5 * (lsgan(critics["RGB"](rgb["rgb"]), True) +
                       lsgan(critics["RGB"](self.rgb_pool.query(saved_rgb)), False))
        d_tir = .5 * (lsgan(critics["TIR"](real_tir), True) +
                       lsgan(critics["TIR"](self.tir_pool.query(saved_tir)), False))
        self._check_losses(dict(D_RGB=d_rgb, D_TIR=d_tir), "discriminator_losses", d_rgb+d_tir)
        (d_rgb + d_tir).backward()
        d_norm = nn.utils.clip_grad_norm_(critics.parameters(), self.config["max_grad_norm"])
        self._require_finite(d_norm, "discriminator_gradients")
        # Validate BOTH graphs before either optimizer changes any weights.
        self.phase = "commit"
        self.scaler.step(self.opt_g)
        self.opt_d.step()
        self.scaler.update()
        self.opt_g.zero_grad(set_to_none=True); self.opt_d.zero_grad(set_to_none=True)
        losses.update(D_RGB=d_rgb.detach(), D_TIR=d_tir.detach(), grad_norm=norm.detach())
        return True, losses

    def state(self):
        return dict(opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(), scaler=self.scaler.state_dict(),
                    noise_rng=self.noise_rng.get_state(), patch_rng=self.patch_rng.get_state(),
                    rgb_pool=self.rgb_pool.state(), tir_pool=self.tir_pool.state(),
                    numerical_failures_total=self.numerical_failures_total)

    @staticmethod
    def validate_state(state):
        for name in ("opt_g", "opt_d"):
            if any(not torch.isfinite(value).all() for item in state[name]["state"].values()
                   for value in item.values() if torch.is_tensor(value)):
                raise ValueError(f"Refuse nonfinite optimizer history: {name}")
        scale = state.get("scaler", {}).get("scale")
        if scale is not None and (not 0 < float(scale) < float("inf")):
            raise ValueError("Refuse invalid AMP scale")

    def restore(self, state):
        self.validate_state(state)
        self.opt_g.load_state_dict(state["opt_g"]); self.opt_d.load_state_dict(state["opt_d"])
        self.scaler.load_state_dict(state["scaler"])
        self.noise_rng.set_state(state["noise_rng"]); self.patch_rng.set_state(state["patch_rng"])
        self.rgb_pool.restore(state["rgb_pool"]); self.tir_pool.restore(state["tir_pool"])
        self.numerical_failures_total = state.get("numerical_failures_total", 0)
