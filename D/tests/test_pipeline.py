"""Exercise real diffusers plumbing with random tiny models; no model download."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from D.common import load_config
from D.preview import generate_one


class TinyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                               StableDiffusionControlNetImg2ImgPipeline,
                               StableDiffusionImg2ImgPipeline, UNet2DConditionModel)
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.temp = tempfile.TemporaryDirectory()
        base = Path(cls.temp.name)
        (base / "vocab.json").write_text(json.dumps({"!": 0, "<|startoftext|>": 1, "<|endoftext|>": 2}))
        (base / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = CLIPTokenizer(str(base / "vocab.json"), str(base / "merges.txt"), model_max_length=77)
        text = CLIPTextModel(CLIPTextConfig(vocab_size=3, hidden_size=16, intermediate_size=32,
                                          num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77,
                                          bos_token_id=1, eos_token_id=2, pad_token_id=2))
        vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1,
                            block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
                            down_block_types=("DownEncoderBlock2D",) * 4,
                            up_block_types=("UpDecoderBlock2D",) * 4, sample_size=64)
        unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                                   layers_per_block=1, block_out_channels=(8, 16), norm_num_groups=4,
                                   down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                   up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                   cross_attention_dim=16, attention_head_dim=4)
        control = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(4, 8, 8, 8))
        # A trained ControlNet has nonzero projections. Activate the tiny ones to test E sensitivity.
        with torch.no_grad():
            for module in [control.controlnet_cond_embedding.conv_out,
                           *control.controlnet_down_blocks, control.controlnet_mid_block]:
                module.weight.normal_(0, .15)
                if module.bias is not None:
                    module.bias.zero_()
        scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False, steps_offset=1)
        shared = dict(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
                      safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        cls.pipeline = StableDiffusionControlNetImg2ImgPipeline(controlnet=control, scheduler=scheduler, **shared)
        cls.plain = StableDiffusionImg2ImgPipeline(scheduler=DDIMScheduler.from_config(scheduler.config), **shared)
        for pipeline in (cls.pipeline, cls.plain):
            pipeline.to("cpu"); pipeline.set_progress_bar_config(disable=True)
        for model in (vae, text, unet, control):
            model.eval().requires_grad_(False)
        cls.config = load_config()
        cls.config.update(width=64, height=64, steps=4, strength=.75, guidance_scale=2.)
        values = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
        cls.initial = Image.fromarray(values).convert("RGB")
        cls.edge = np.zeros((64, 64), np.float32); cls.edge[:, 22:25] = .65

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_d0_matches_plain_img2img_with_identical_noise_and_settings(self):
        config = self.config
        d0, _ = generate_one(self.pipeline, self.initial, self.edge, config, "D0")
        with torch.inference_mode():
            plain = self.plain(prompt=config["prompt"], negative_prompt=config["negative_prompt"],
                               image=self.initial, num_inference_steps=config["steps"],
                               strength=config["strength"], guidance_scale=config["guidance_scale"],
                               eta=0., generator=torch.Generator("cpu").manual_seed(42)).images[0]
        np.testing.assert_array_equal(np.asarray(d0), np.asarray(plain))

    def test_d1_repeats_exactly_and_responds_to_the_soft_condition(self):
        first, _ = generate_one(self.pipeline, self.initial, self.edge, self.config, "D1")
        second, _ = generate_one(self.pipeline, self.initial, self.edge, self.config, "D1")
        blank, _ = generate_one(self.pipeline, self.initial, self.edge * 0, self.config, "D1")
        d0, _ = generate_one(self.pipeline, self.initial, self.edge, self.config, "D0")
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertTrue(np.any(np.asarray(first) != np.asarray(blank)))
        self.assertTrue(np.any(np.asarray(first) != np.asarray(d0)))


if __name__ == "__main__":
    unittest.main()
