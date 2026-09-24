"""CycleGAN with C0-cap zero conditions or C1 hard Canny edges."""

import itertools

import torch

from .conditioned_generator import ConditionedGenerator
from .cycle_gan_model import CycleGANModel


class ConditionedCycleGANModel(CycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--condition_mode",
            choices=("zero", "canny"),
            default="zero",
            help="zero is C0-cap; canny supplies a hard TIR edge map for C1.",
        )
        parser.add_argument("--edge_sigma", type=float, default=1.0)
        parser.add_argument("--edge_low_threshold", type=float, default=0.08)
        parser.add_argument("--edge_high_threshold", type=float, default=0.16)
        return parser

    def __init__(self, opt):
        if opt.direction != "BtoA" or opt.input_nc != 1 or opt.output_nc != 3:
            raise ValueError("Conditioned CycleGAN requires --direction BtoA --input_nc 1 --output_nc 3")
        if opt.isTrain and opt.lambda_identity != 0:
            raise ValueError("Conditioned CycleGAN requires --lambda_identity 0, matching the C0 baseline")

        super().__init__(opt)
        self.netG_A = ConditionedGenerator(
            self.netG_A,
            opt.condition_mode,
            edge_sigma=opt.edge_sigma,
            edge_low_threshold=opt.edge_low_threshold,
            edge_high_threshold=opt.edge_high_threshold,
        )

        if self.isTrain:
            # The parent optimizer was constructed before G_A was wrapped.
            # Rebuild it so the adapter receives gradient updates too.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers[0] = self.optimizer_G

    def setup(self, opt):
        super().setup(opt)
        if self.isTrain and not opt.continue_train:
            # BaseModel.setup initializes the whole wrapper. Restore the
            # adapter's identity start only for a fresh training run.
            generator = self.netG_A.module if hasattr(self.netG_A, "module") else self.netG_A
            generator.reset_adapter_to_identity()
