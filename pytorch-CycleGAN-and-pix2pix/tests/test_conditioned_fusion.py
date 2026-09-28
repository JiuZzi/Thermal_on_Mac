"""Small CPU checks for the legacy and direct condition interfaces."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.conditioned_cycle_gan_model import ConditionedCycleGANModel
from models.conditioned_generator import ConditionedGenerator


class ConditionedFusionTests(unittest.TestCase):
    @staticmethod
    def thermal_step():
        thermal = torch.full((1, 1, 32, 32), -1.0)
        thermal[:, :, :, 16:] = 1.0
        return thermal

    def test_legacy_adapter_is_still_the_default(self):
        generator = ConditionedGenerator(nn.Identity(), condition_mode="zero")
        generator.reset_adapter_to_identity()
        thermal = self.thermal_step()

        torch.testing.assert_close(generator(thermal), thermal)
        self.assertIn("adapter.weight", generator.state_dict())

    def test_direct_zero_mode_keeps_three_separate_channels(self):
        generator = ConditionedGenerator(nn.Identity(), condition_mode="zero", fusion_mode="direct")
        output = generator(self.thermal_step())

        self.assertEqual(output.shape, (1, 3, 32, 32))
        torch.testing.assert_close(output[:, :1], self.thermal_step())
        self.assertEqual(torch.count_nonzero(output[:, 1:]).item(), 0)
        self.assertNotIn("adapter.weight", generator.state_dict())

    def test_direct_canny_channel_is_nonzero_and_trainable(self):
        backbone = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        generator = ConditionedGenerator(backbone, condition_mode="canny", fusion_mode="direct")
        thermal = self.thermal_step()
        edge, trusted_edge = generator.make_condition(thermal)

        self.assertGreater(torch.count_nonzero(edge).item(), 0)
        self.assertEqual(torch.count_nonzero(trusted_edge).item(), 0)
        generator(thermal).sum().backward()
        self.assertGreater(backbone.weight.grad[:, 1].abs().sum().item(), 0)

    def test_direct_model_optimizer_uses_three_channel_backbone(self):
        opt = SimpleNamespace(
            isTrain=True,
            direction="BtoA",
            input_nc=1,
            output_nc=3,
            lambda_identity=0,
            lambda_A=10.0,
            lambda_B=10.0,
            checkpoints_dir="unused",
            name="direct_fusion_test",
            device=torch.device("cpu"),
            preprocess="crop",
            seed=42,
            ngf=8,
            ndf=8,
            netG="resnet_6blocks",
            netD="basic",
            norm="instance",
            no_dropout=True,
            init_type="normal",
            init_gain=0.02,
            n_layers_D=3,
            pool_size=0,
            gan_mode="lsgan",
            lr=0.0002,
            beta1=0.5,
            condition_mode="canny",
            fusion_mode="direct",
            edge_sigma=1.0,
            edge_low_threshold=0.08,
            edge_high_threshold=0.16,
        )
        model = ConditionedCycleGANModel(opt)
        first_conv = model.netG_A.backbone.model[1]

        self.assertEqual(first_conv.in_channels, 3)
        self.assertTrue(any(first_conv.weight is param for group in model.optimizer_G.param_groups for param in group["params"]))

        model.set_input(
            {
                "A": torch.rand((1, 3, 32, 32)) * 2 - 1,
                "B": self.thermal_step(),
                "A_paths": ["rgb.png"],
                "B_paths": ["thermal.png"],
            }
        )
        model.optimize_parameters()

        self.assertEqual(model.fake_B.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(model.loss_G).item())
        self.assertGreater(first_conv.weight.grad[:, 1].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
