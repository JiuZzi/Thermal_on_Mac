"""Opt-in learned RGB colors; the existing manual-color model stays unchanged."""

import hashlib
import itertools
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from .learned_color import LearnedColorGenerator
from .guidance_diagnostics import guidance_report, save_report
from .semantic_edges import FMB_CLASS_NAMES
from .semantic_color_cycle_gan_model import SemanticColorCycleGANModel


class LearnedColorCycleGANModel(SemanticColorCycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SemanticColorCycleGANModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(color_condition="semantic", lambda_color_prior=0.)
        parser.add_argument("--learned_color_prior", required=True, help="Frozen RGB-learned JSON artifact")
        parser.add_argument("--learned_color_mode", choices=("learned", "zero_tokens"), default="learned")
        parser.add_argument("--learned_color_width", type=int, default=16)
        parser.add_argument("--learned_color_dropout", type=float, default=.1)
        parser.add_argument("--learned_color_weight_decay", type=float, default=.0001)
        parser.add_argument("--learned_color_loss_weight", type=float, default=.01,
                            help="Bounded learned GMM chroma penalty; 0 restores conditioning-only training")
        parser.add_argument("--learned_color_loss_min_quality", type=float, default=.5)
        parser.add_argument("--learned_color_loss_warmup_steps", type=int, default=1000,
                            help="Ramp color weight over optimizer updates; 0 disables the ramp")
        parser.add_argument("--reliability_flip", action="store_true")
        parser.add_argument("--reliability_boundary_radius", type=int, default=1)
        parser.add_argument("--gate_semantic_conditions", action="store_true")
        parser.add_argument("--init_condition_dir", type=str, default=None,
                            help="Semantic-only G_A/G_B warm-start, fresh optimizer/discriminators")
        parser.add_argument("--init_condition_epoch", default="5")
        parser.add_argument("--color_feature_override", choices=("normal", "off", "shuffle"), default="normal",
                            help="Test diagnostic only; does not change the checkpoint identity")
        return parser

    def __init__(self, opt):
        if opt.color_condition != "semantic" or opt.lambda_color_prior != 0:
            raise ValueError("Manual chroma loss must be off; use learned_color_loss_weight for the learned GMM loss")
        if (not math.isfinite(opt.learned_color_loss_weight) or opt.learned_color_loss_weight < 0
                or not math.isfinite(opt.learned_color_loss_min_quality) or not 0 < opt.learned_color_loss_min_quality <= 1
                or opt.learned_color_loss_warmup_steps < 0):
            raise ValueError("Invalid learned color loss weight, reliability threshold or warmup")
        if (not 0 <= opt.learned_color_weight_decay < float("inf")
                or not 0 <= opt.reliability_boundary_radius <= 8):
            raise ValueError("Invalid regularization/reliability settings")
        if opt.init_condition_dir and (not opt.isTrain or opt.continue_train or opt.init_semantic_dir):
            raise ValueError("Use one warm-start source, only for fresh training")
        if opt.isTrain and opt.color_feature_override != "normal":
            raise ValueError("Feature perturbations are test diagnostics only")
        super().__init__(opt)
        self.netG_A = LearnedColorGenerator(self.netG_A, opt.learned_color_prior,
            width=opt.learned_color_width, mode=opt.learned_color_mode,
            flip=opt.reliability_flip, boundary_radius=opt.reliability_boundary_radius,
            gate_semantic=opt.gate_semantic_conditions, dropout=opt.learned_color_dropout,
            loss_weight=opt.learned_color_loss_weight, loss_min_quality=opt.learned_color_loss_min_quality,
            loss_warmup_steps=opt.learned_color_loss_warmup_steps)
        self.netG_A.feature_override = opt.color_feature_override
        self.visual_names.append("semantic_quality")
        self.loss_names.extend(["color_prior_raw", "color_coverage", "color_weight"])
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam([
                {"params": itertools.chain(self.netG_A.semantic.parameters(), self.netG_B.parameters())},
                {"params": itertools.chain(self.netG_A.learned_encoder.parameters(), self.netG_A.learned_projections.parameters()),
                 "weight_decay": opt.learned_color_weight_decay}], lr=opt.lr, betas=(opt.beta1, .999))
            self.optimizers[0] = self.optimizer_G

    def setup(self, opt):
        super().setup(opt)
        generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
        metadata = dict(generator.color_metadata, quality_evidence="implementation only; external generalization unverified",
                        feature_override=opt.color_feature_override, weight_decay=opt.learned_color_weight_decay,
                        initialization="loaded checkpoint" if not self.isTrain or opt.continue_train else "fresh")
        if self.isTrain and not opt.continue_train:
            if opt.init_condition_dir:
                root = Path(opt.init_condition_dir).expanduser().resolve()
                paths = {name: root / f"{opt.init_condition_epoch}_net_{name}.pth" for name in ("G_A", "G_B")}
                generator.semantic.load_state_dict(torch.load(paths["G_A"], map_location="cpu", weights_only=True), strict=True)
                self.netG_B.load_state_dict(torch.load(paths["G_B"], map_location="cpu", weights_only=True), strict=True)
                generator.reset_learned_fusion()
                metadata.update(initialization="semantic-only G_A/G_B; fresh D/optimizers",
                                sources={name: dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in paths.items()})
            elif opt.init_semantic_dir:
                metadata["initialization"] = "legacy contour G_A/G_B; fresh D/optimizers"
        if not dist.is_initialized() or dist.get_rank() == 0:
            (self.save_dir / ("color_train.json" if self.isTrain else "color_test.json")).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            print(f"Learned GMM color loss: max weight {opt.learned_color_loss_weight}; "
                  f"Q >= {opt.learned_color_loss_min_quality}; warmup {opt.learned_color_loss_warmup_steps} updates; "
                  "manual color loss disabled")

    def forward(self):
        super().forward()
        self.semantic_quality = self.color_aux["quality"].detach()*2-1

    def save_guidance_diagnostics(self, epoch, total_iters):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
        supported = [name for name, flag in zip(FMB_CLASS_NAMES,
                     generator.learned_prior.supported.detach().cpu().tolist()) if flag]
        report = guidance_report(self.color_aux["probabilities"], self.color_aux["quality"], FMB_CLASS_NAMES,
            supported=supported, min_quality=self.opt.learned_color_loss_min_quality,
            boundary_radius=generator.boundary_radius, feature_gate=self.color_aux["feature_gate"],
            scope="latest logged real-TIR minibatch; not an epoch average")
        report.update(epoch=epoch, samples_seen=total_iters, flip_reliability=generator.flip,
                      semantic_gate_enabled=generator.gate_semantic,
                      color_loss_enabled=self.opt.learned_color_loss_weight > 0,
                      contour_variant=generator.semantic.contour_variant)
        save_report(report, self.save_dir)

    def backward_G(self):
        self.loss_idt_A = self.loss_idt_B = 0
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)*self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)*self.opt.lambda_B
        self.loss_color_prior_raw = self.fake_B.sum()*0
        self.loss_color_coverage = 0.
        self.loss_color_weight = 0.
        if self.opt.learned_color_loss_weight > 0:
            generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
            self.loss_color_prior_raw, self.loss_color_coverage = generator.learned_prior.chroma_loss(
                self.fake_B, self.color_aux["probabilities"], self.color_aux["quality"],
                min_quality=self.opt.learned_color_loss_min_quality)
            with torch.no_grad():
                generator.color_loss_updates.add_(1)
            warmup = self.opt.learned_color_loss_warmup_steps
            ramp = min(1., int(generator.color_loss_updates)/warmup) if warmup else 1.
            self.loss_color_weight = self.opt.learned_color_loss_weight*ramp
        self.loss_color_prior = self.loss_color_prior_raw*self.loss_color_weight
        self.loss_G = self.loss_G_A+self.loss_G_B+self.loss_cycle_A+self.loss_cycle_B+self.loss_color_prior
        self.loss_G.backward()
