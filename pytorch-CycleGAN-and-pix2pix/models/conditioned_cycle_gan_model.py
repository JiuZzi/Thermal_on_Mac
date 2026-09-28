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


class ConditionedCycleGANModel(CycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--condition_mode",
            choices=("zero", "canny", "soft_single", "soft_multi"),
            default="zero",
            help="zero=C0-cap, canny=C1, soft_single=C2a, soft_multi=C2b.",
        )
        parser.add_argument("--edge_sigma", type=float, default=1.0)
        parser.add_argument("--edge_low_threshold", type=float, default=0.08)
        parser.add_argument("--edge_high_threshold", type=float, default=0.16)
        parser.add_argument("--edge_soft_width", type=float, default=1.0,
                            help="Gaussian distance-decay width in pixels for C2a/C2b.")
        parser.add_argument("--soft_high_thresholds", type=parse_soft_high_thresholds,
                            default=(0.12, 0.16, 0.20),
                            help="Increasing high thresholds for C2b, separated by commas; equal weights.")
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
        if opt.condition_mode in ("soft_single", "soft_multi") and fusion_mode != "direct":
            raise ValueError("C2a/C2b require --fusion_mode direct to match the direct-fusion C1 control")
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
