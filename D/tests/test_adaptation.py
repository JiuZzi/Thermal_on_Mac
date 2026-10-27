import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from D.adaptation import configure_trainable, partial_state, noise_prediction, save_adapter, load_adapter
from D.prepare_rgb import CITY_TO_FMB, NAMES, map_probabilities
from D.tests import test_pipeline as tiny_fixture


class AdaptationTests(unittest.TestCase):
    def test_rgb_probability_mapping_preserves_mass_and_important_categories(self):
        names = list(CITY_TO_FMB)
        values = np.full((19, 3, 4), 1/19, np.float32)
        mapped = map_probabilities(values, names)
        np.testing.assert_allclose(mapped.sum(0), 1, atol=1e-6)
        np.testing.assert_allclose(mapped[NAMES.index("person")], 2/19)
        np.testing.assert_allclose(mapped[NAMES.index("building")], 1/19)
        with self.assertRaises(ValueError):
            map_probabilities(values, ["unknown"] * 19)

    def test_gradients_cross_frozen_unet_and_only_adapter_weights_update(self):
        tiny_fixture.TinyPipelineTests.setUpClass()
        try:
            pipeline = tiny_fixture.TinyPipelineTests.pipeline
            controlnet, unet = pipeline.controlnet, pipeline.unet
            frozen_unet = {n:p.detach().clone() for n,p in unet.named_parameters()}
            frozen_body = controlnet.conv_in.weight.detach().clone()
            parameters = configure_trainable(controlnet, unet)
            before = {n:v.clone() for n,v in partial_state(controlnet).items()}
            optimizer = torch.optim.AdamW(parameters, lr=.001)
            rng = torch.Generator("cpu").manual_seed(42)
            noisy = torch.randn((1,4,8,8), generator=rng)
            text = torch.randn((1,77,16), generator=rng)
            condition = torch.rand((1,3,64,64), generator=rng)
            target = torch.randn(noisy.shape, generator=rng)
            loss = F.mse_loss(noise_prediction(controlnet, unet, noisy, torch.tensor([50]), text, condition), target)
            loss.backward()
            self.assertGreater(float(controlnet.controlnet_cond_embedding.conv_in.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in unet.parameters()))
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[n],v) for n,v in partial_state(controlnet).items()))
            self.assertTrue(all(torch.equal(p.detach(), frozen_unet[n]) for n,p in unet.named_parameters()))
            self.assertTrue(torch.equal(controlnet.conv_in.weight, frozen_body))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "adapter.safetensors"
                save_adapter(controlnet, path)
                saved = {n:v.clone() for n,v in partial_state(controlnet).items()}
                with torch.no_grad():
                    controlnet.controlnet_cond_embedding.conv_in.weight.zero_()
                load_adapter(controlnet, path)
                self.assertTrue(all(torch.equal(saved[n],v) for n,v in partial_state(controlnet).items()))
        finally:
            tiny_fixture.TinyPipelineTests.tearDownClass()


if __name__ == "__main__":
    unittest.main()
