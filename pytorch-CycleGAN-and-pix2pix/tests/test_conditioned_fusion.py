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

    def test_soft_single_preserves_hard_edges_and_is_continuous(self):
        thermal = self.thermal_step()
        hard_model = ConditionedGenerator(nn.Identity(), condition_mode="canny", fusion_mode="direct")
        soft_model = ConditionedGenerator(nn.Identity(), condition_mode="soft_single", fusion_mode="direct")
        hard, _ = hard_model.make_condition(thermal)
        soft, reserved = soft_model.make_condition(thermal)

        self.assertEqual(soft.shape, thermal.shape)
        self.assertTrue(torch.all((soft >= 0) & (soft <= 1)))
        torch.testing.assert_close(soft[hard.bool()], torch.ones_like(soft[hard.bool()]))
        self.assertTrue(torch.any((soft > 0) & (soft < 1)).item())
        self.assertEqual(torch.count_nonzero(reserved).item(), 0)

    def test_soft_multi_is_equal_mean_of_softened_thresholds(self):
        thermal = self.thermal_step()
        thresholds = (0.12, 0.16, 0.20)
        multi = ConditionedGenerator(
            nn.Identity(), condition_mode="soft_multi", fusion_mode="direct",
            soft_high_thresholds=thresholds,
        )
        expected = torch.stack([
            ConditionedGenerator(
                nn.Identity(), condition_mode="soft_single", fusion_mode="direct",
                edge_low_threshold=high / 2, edge_high_threshold=high,
            ).make_condition(thermal)[0]
            for high in thresholds
        ]).mean(0)
        actual, _ = multi.make_condition(thermal)

        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)

    def test_soft_modes_handle_empty_edges_and_validate_configuration(self):
        flat = torch.zeros_like(self.thermal_step())
        for mode in ("soft_single", "soft_multi"):
            generator = ConditionedGenerator(nn.Identity(), condition_mode=mode, fusion_mode="direct")
            edge, _ = generator.make_condition(flat)
            self.assertEqual(torch.count_nonzero(edge).item(), 0)
        with self.assertRaisesRegex(ValueError, "width"):
            ConditionedGenerator(nn.Identity(), condition_mode="soft_single", edge_soft_width=0)
        with self.assertRaisesRegex(ValueError, "include the C1"):
            ConditionedGenerator(
                nn.Identity(), condition_mode="soft_multi", soft_high_thresholds=(0.10, 0.20),
            )

    def test_soft_modes_reject_legacy_adapter_in_cycle_gan(self):
        opt = SimpleNamespace(
            direction="BtoA", input_nc=1, output_nc=3, isTrain=False,
            condition_mode="soft_single", fusion_mode="adapter",
            checkpoints_dir="unused", name="soft_adapter_error", device=torch.device("cpu"),
            preprocess="crop", ngf=8, netG="resnet_6blocks", norm="instance",
            no_dropout=True, init_type="normal", init_gain=0.02,
        )
        with self.assertRaisesRegex(ValueError, "fusion_mode direct"):
            ConditionedCycleGANModel(opt)

    def test_soft_edge_override_can_measure_generator_reliance(self):
        backbone = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            backbone.weight.zero_()
            backbone.weight[0, 1, 0, 0] = 1
        generator = ConditionedGenerator(backbone, condition_mode="soft_single", fusion_mode="direct")
        thermal = self.thermal_step()
        normal = generator(thermal)
        without_edge = generator(thermal, edge_override=torch.zeros_like(thermal))

        self.assertGreater((normal - without_edge).abs().mean().item(), 0)
        self.assertEqual(torch.count_nonzero(without_edge).item(), 0)
        with self.assertRaisesRegex(ValueError, "shape"):
            generator(thermal, edge_override=torch.zeros((1, 2, 32, 32)))

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
