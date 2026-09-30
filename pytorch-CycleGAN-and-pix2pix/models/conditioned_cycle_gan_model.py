"""CycleGAN with zero, hard Canny, or fixed soft Canny conditions."""

import argparse
import itertools

import torch

from .conditioned_generator import ConditionedGenerator
from .cycle_gan_model import CycleGANModel
from . import networks


def parse_soft_high_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated Canny high thresholds") from error
    if len(thresholds) < 2 or any(not (0 < threshold <= 1) for threshold in thresholds):
        raise argparse.ArgumentTypeError("Provide at least two Canny high thresholds in (0, 1]")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise argparse.ArgumentTypeError("Canny high thresholds must be unique and increasing")
    return thresholds


def parse_saliency_sigmas(value: str) -> tuple[float, float, float]:
    try:
        sigmas = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected three comma-separated smoothing scales") from error
    if len(sigmas) != 3 or any(sigma <= 0 for sigma in sigmas) or tuple(sorted(sigmas)) != sigmas:
        raise argparse.ArgumentTypeError("Provide three positive, increasing smoothing scales")
    return sigmas


class ConditionedCycleGANModel(CycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--condition_mode",
            choices=("zero", "canny", "soft_single", "soft_multi", "soft_saliency", "soft_saliency_position", "soft_object_line"),
            default="zero",
            help="zero=C0-cap, canny=C1, soft_single=C2a, soft_multi=C2b, soft_saliency=adaptive C2c, soft_saliency_position=former position C2c, soft_object_line=detector/line candidate.",
        )
        parser.add_argument("--edge_sigma", type=float, default=1.0)
        parser.add_argument("--edge_low_threshold", type=float, default=0.08)
        parser.add_argument("--edge_high_threshold", type=float, default=0.16)
        parser.add_argument("--edge_soft_width", type=float, default=1.0,
                            help="Gaussian distance-decay width in pixels for C2a/C2b.")
        parser.add_argument("--soft_high_thresholds", type=parse_soft_high_thresholds,
                            default=(0.12, 0.16, 0.20),
                            help="Increasing high thresholds for C2b, separated by commas; equal weights.")
        parser.add_argument("--saliency_sigmas", type=parse_saliency_sigmas, default=(0.7, 1.0, 1.6),
                            help="C2c fine, middle, and coarse Canny scales.")
        parser.add_argument("--saliency_inner_weight", type=float, default=0.8,
                            help="Former position C2c fine-candidate weight inside the middle third.")
        parser.add_argument("--saliency_outer_weight", type=float, default=0.2,
                            help="Former position C2c fine-candidate weight outside the middle third.")
        parser.add_argument("--saliency_transition_fraction", type=float, default=24.0 / 256.0,
                            help="Former position C2c transition width as a fraction of image height.")
        parser.add_argument("--saliency_background_gain", type=float, default=0.5,
                            help="C2c gain for coarse edges outside important regions.")
        parser.add_argument("--detector_checkpoint", type=str, default=None,
                            help="Frozen FLIR detector checkpoint required for soft_object_line.")
        parser.add_argument("--detector_score_threshold", type=float, default=0.20,
                            help="Object detection score threshold for soft_object_line.")
        parser.add_argument(
            "--fusion_mode",
            choices=("adapter", "direct"),
            default="adapter",
            help="adapter preserves old 3-to-1 checkpoints; direct feeds all three condition channels into G_A.",
        )
        return parser

    def __init__(self, opt):
        if opt.direction != "BtoA" or opt.input_nc != 1 or opt.output_nc != 3:
            raise ValueError("Conditioned CycleGAN requires --direction BtoA --input_nc 1 --output_nc 3")
        if opt.isTrain and opt.lambda_identity != 0:
            raise ValueError("Conditioned CycleGAN requires --lambda_identity 0, matching the C0 baseline")

        super().__init__(opt)
        fusion_mode = getattr(opt, "fusion_mode", "adapter")
        if opt.condition_mode in ("soft_single", "soft_multi", "soft_saliency", "soft_saliency_position", "soft_object_line") and fusion_mode != "direct":
            raise ValueError("C2a/C2b/C2c require --fusion_mode direct to match the direct-fusion C1 control")
        backbone = self.netG_A
        if fusion_mode == "direct":
            backbone = networks.define_G(
                3, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                not opt.no_dropout, opt.init_type, opt.init_gain,
            )
        self.netG_A = ConditionedGenerator(
            backbone,
            opt.condition_mode,
            edge_sigma=opt.edge_sigma,
            edge_low_threshold=opt.edge_low_threshold,
            edge_high_threshold=opt.edge_high_threshold,
            edge_soft_width=getattr(opt, "edge_soft_width", 1.0),
            soft_high_thresholds=getattr(opt, "soft_high_thresholds", (0.12, 0.16, 0.20)),
            fusion_mode=fusion_mode,
            saliency_sigmas=getattr(opt, "saliency_sigmas", (0.7, 1.0, 1.6)),
            saliency_inner_weight=getattr(opt, "saliency_inner_weight", 0.8),
            saliency_outer_weight=getattr(opt, "saliency_outer_weight", 0.2),
            saliency_transition_fraction=getattr(opt, "saliency_transition_fraction", 24.0 / 256.0),
            saliency_background_gain=getattr(opt, "saliency_background_gain", 0.5),
            detector_checkpoint=getattr(opt, "detector_checkpoint", None),
            detector_score_threshold=getattr(opt, "detector_score_threshold", 0.20),
        )

        if self.isTrain:
            # The parent optimizer was constructed before G_A was wrapped.
            # Rebuild it for the selected backbone and optional adapter.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers[0] = self.optimizer_G

    def setup(self, opt):
        super().setup(opt)
        generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
        if self.isTrain and not opt.continue_train and generator.fusion_mode == "adapter":
            # BaseModel.setup initializes the whole wrapper. Restore the
            # adapter's identity start only for a fresh training run.
            generator.reset_adapter_to_identity()
