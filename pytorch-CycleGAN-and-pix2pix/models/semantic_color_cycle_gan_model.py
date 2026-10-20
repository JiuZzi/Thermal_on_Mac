"""Opt-in full semantics + color-family conditioning, with joint512 unchanged."""

import hashlib
import itertools
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .conditioned_cycle_gan_model import ConditionedCycleGANModel
from .semantic_color import CONDITION_MODES, SemanticColorGenerator
from .semantic_contour_cycle_gan_model import SemanticContourCycleGANModel


class SemanticColorCycleGANModel(SemanticContourCycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SemanticContourCycleGANModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--color_condition", choices=CONDITION_MODES, default="semantic_color")
        parser.add_argument("--color_width", type=int, default=16)
        parser.add_argument("--lambda_color_prior", type=float, default=None,
                            help="Default 1 for semantic_color, otherwise 0; experimental chroma weight")
        parser.add_argument("--init_semantic_dir", type=str, default=None,
                            help="Optional old semantic_contour checkpoint directory; fresh training only")
        parser.add_argument("--init_semantic_epoch", default="80")
        return parser

    def __init__(self, opt):
        if opt.condition_mode != "soft_saliency" or opt.fusion_mode != "direct":
            raise ValueError("Semantic/color conditioning requires the unchanged direct joint512")
        if opt.netG not in ("resnet_6blocks", "resnet_9blocks"):
            raise ValueError("Semantic/color prototype supports resnet_6blocks or resnet_9blocks")
        self.color_weight = (1. if opt.color_condition == "semantic_color" else 0.) if opt.lambda_color_prior is None else opt.lambda_color_prior
        if not 0 <= self.color_weight < float("inf"):
            raise ValueError("lambda_color_prior must be finite and nonnegative")
        if opt.color_condition != "semantic_color" and self.color_weight:
            raise ValueError("Color loss requires color_condition=semantic_color")
        if opt.init_semantic_dir and (not opt.isTrain or opt.continue_train):
            raise ValueError("Legacy initialization is only for a fresh training run")
        opt.lambda_color_prior = self.color_weight
        ConditionedCycleGANModel.__init__(self, opt)
        self.netG_A = SemanticColorGenerator(
            self.netG_A.backbone, opt.semantic_checkpoint,
            color_condition=opt.color_condition, color_width=opt.color_width,
            edge_sigma=opt.edge_sigma, edge_low_threshold=opt.edge_low_threshold,
            edge_high_threshold=opt.edge_high_threshold, edge_soft_width=opt.edge_soft_width,
            soft_high_thresholds=opt.soft_high_thresholds, saliency_sigmas=opt.saliency_sigmas,
            saliency_background_gain=opt.saliency_background_gain, saliency_variant=opt.saliency_variant,
        )
        self.loss_names.append("color_prior")
        self.visual_names.extend(("contour_joint", "color_hint", "color_strength"))
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, .999),
            )
            self.optimizers[0] = self.optimizer_G

    def setup(self, opt):
        super().setup(opt)
        generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
        metadata = dict(generator.color_metadata, lambda_color_prior=self.color_weight,
                        initialization="fresh" if self.isTrain and not opt.continue_train else "loaded checkpoint",
                        quality_evidence="implementation checks do not establish color or structure gains")
        if self.isTrain and not opt.continue_train:
            generator.reset_color_fusion()  # BaseModel.setup reinitializes all Conv layers.
            if opt.init_semantic_dir:
                root = Path(opt.init_semantic_dir).expanduser().resolve()
                sources = {name: root / f"{opt.init_semantic_epoch}_net_{name}.pth" for name in ("G_A", "G_B")}
                states = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in sources.items()}
                generator.initialize_from_semantic(states["G_A"])
                reverse = self.netG_B.module if hasattr(self.netG_B, "module") else self.netG_B
                reverse.load_state_dict(states["G_B"], strict=True)
                metadata.update(initialization="legacy G_A/G_B; fresh discriminators and optimizer",
                                sources={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                         for name, path in sources.items()})
        if not dist.is_initialized() or dist.get_rank() == 0:
            filename = "color_train.json" if self.isTrain else "color_test.json"
            (self.save_dir / filename).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Independent condition: {opt.color_condition}; color loss weight: {self.color_weight}; third channel remains zero")

    def forward(self):
        self.fake_B, self.color_aux = self.netG_A(self.real_A, return_aux=True)
        self.contour_joint = self.color_aux["edge"].detach() * 2 - 1
        self.color_hint = self.color_aux["color"]["hint_rgb"].detach() * 2 - 1
        self.color_strength = self.color_aux["color"]["strength"].detach() * 2 - 1
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_G(self):
        self.loss_idt_A = self.loss_idt_B = 0  # The inherited conditioned interface requires identity=0.
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
        self.loss_color_prior = generator.color_prior.chroma_loss(self.fake_B, self.color_aux["color"]) * self.color_weight
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_color_prior)
        self.loss_G.backward()
