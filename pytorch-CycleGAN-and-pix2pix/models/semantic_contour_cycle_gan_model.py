"""CycleGAN with frozen online structure + semantic protected soft contours."""

import itertools
import json

import torch
import torch.distributed as dist

from .conditioned_cycle_gan_model import ConditionedCycleGANModel
from .semantic_contour import DEFAULT_CHECKPOINT, SemanticContourGenerator
from .selective_semantic_edges import VARIANTS


class SemanticContourCycleGANModel(ConditionedCycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = ConditionedCycleGANModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(condition_mode="soft_saliency", fusion_mode="direct")
        if is_train:
            parser.set_defaults(lambda_identity=0, n_epochs=40, n_epochs_decay=40, seed=42)
        parser.add_argument("--semantic_checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                            help="Complete frozen FMB SegFormer-B3 best_model.pth; required in training and testing")
        parser.add_argument("--contour_variant", choices=VARIANTS, default="joint512",
                            help="Keep joint512 for existing experiments; selective_v1/clarity_v1/recall_v1 are separate candidates")
        return parser

    def __init__(self, opt):
        if opt.condition_mode != "soft_saliency" or opt.fusion_mode != "direct":
            raise ValueError("semantic_contour_cycle_gan requires soft_saliency and direct fusion")
        super().__init__(opt)
        self.netG_A = SemanticContourGenerator(
            self.netG_A.backbone, opt.semantic_checkpoint,
            contour_variant=getattr(opt, "contour_variant", "joint512"),
            edge_sigma=opt.edge_sigma, edge_low_threshold=opt.edge_low_threshold,
            edge_high_threshold=opt.edge_high_threshold, edge_soft_width=opt.edge_soft_width,
            soft_high_thresholds=opt.soft_high_thresholds,
            saliency_sigmas=opt.saliency_sigmas,
            saliency_background_gain=opt.saliency_background_gain,
            saliency_variant=getattr(opt, "saliency_variant", "structure_texture"),
        )
        self.visual_names.append("contour_joint")
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999),
            )
            self.optimizers[0] = self.optimizer_G

    def setup(self, opt):
        super().setup(opt)
        generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
        print(f"Second channel: {generator.semantic_metadata.get('contour_variant', 'joint512')}; frozen SegFormer-B3 at 512; third channel: zero")
        print("Semantic checkpoint SHA256:", generator.semantic_metadata["checkpoint_sha256"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            path = self.save_dir / ("semantic_train.json" if self.isTrain else "semantic_test.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(generator.semantic_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def forward(self):
        self.fake_B, contour = self.netG_A(self.real_A, return_condition=True)
        self.contour_joint = contour.detach() * 2 - 1  # visualizer encoding only
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)
