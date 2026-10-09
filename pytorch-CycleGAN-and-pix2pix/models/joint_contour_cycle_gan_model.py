"""Single-stage label-free contour refinement + CycleGAN, with/without NCE."""

import copy
import hashlib
import itertools
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from . import networks
from .conditioned_cycle_gan_model import ConditionedCycleGANModel
from .joint_contour import (
    JointContourGenerator, anchor_loss, heuristic_anchors,
    local_map_contrast, transform_view,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(state, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class JointContourCycleGANModel(ConditionedCycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = ConditionedCycleGANModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(condition_mode="soft_saliency", fusion_mode="direct")
        if is_train:
            parser.set_defaults(lambda_identity=0, n_epochs=40, n_epochs_decay=40, seed=2026)
        parser.add_argument("--joint_variant", choices=("plain", "contrastive"), default="plain")
        parser.add_argument("--contour_width", type=int, default=16)
        parser.add_argument("--contour_embedding_dim", type=int, default=16)
        parser.add_argument("--contour_epsilon", type=float, default=0.02)
        parser.add_argument("--contour_residual_limit", type=float, default=6.0)
        parser.add_argument("--lambda_contour_anchor", type=float, default=1.0)
        parser.add_argument("--lambda_contour_nce", type=float, default=0.1)
        parser.add_argument("--contour_positive_threshold", type=float, default=0.6)
        parser.add_argument("--contour_background_threshold", type=float, default=0.02)
        parser.add_argument("--contour_gradient_threshold", type=float, default=0.02)
        parser.add_argument("--contour_exclusion_radius", type=int, default=3)
        parser.add_argument("--contour_max_anchors", type=int, default=64)
        parser.add_argument("--contour_max_negatives", type=int, default=128)
        parser.add_argument("--contour_temperature", type=float, default=0.1)
        parser.add_argument("--contour_intensity_jitter", type=float, default=0.05)
        parser.add_argument("--contour_sampling_seed", type=int, default=2026)
        parser.add_argument("--joint_smoke", action="store_true", help="Mark short implementation checks; not quality evidence")
        parser.add_argument("--contour_intervention", choices=("normal", "zero", "shift", "prior"), default="normal",
                            help="Inference-only causal checks of the contour channel")
        return parser

    def __init__(self, opt):
        if dist.is_initialized():
            raise ValueError("This first joint implementation supports single-device training only")
        if opt.fusion_mode != "direct" or opt.condition_mode not in ("soft_multi", "soft_saliency"):
            raise ValueError("Use direct fusion and soft_multi/soft_saliency")
        if opt.isTrain and (opt.contour_intervention != "normal" or opt.seed is None):
            raise ValueError("Training requires normal intervention and an explicit random seed")
        if opt.lambda_contour_anchor <= 0 or opt.lambda_contour_nce < 0:
            raise ValueError("Anchor weight must be positive; NCE weight must be nonnegative")
        if opt.joint_variant == "contrastive" and opt.lambda_contour_nce <= 0:
            raise ValueError("Contrastive variant requires positive lambda_contour_nce")
        if not 0 <= opt.contour_background_threshold < opt.contour_positive_threshold <= 1:
            raise ValueError("Invalid heuristic anchor thresholds")
        if opt.contour_gradient_threshold < 0 or opt.contour_exclusion_radius < 1:
            raise ValueError("Gradient threshold >= 0 and exclusion radius >= 1 required")
        if min(opt.contour_max_anchors, opt.contour_max_negatives) < 1 or opt.contour_temperature <= 0:
            raise ValueError("Invalid NCE sampling/temperature")
        if not 0 <= opt.contour_intensity_jitter <= 0.2 or opt.contour_sampling_seed < 0:
            raise ValueError("Invalid augmentation/seed settings")
        if opt.isTrain and opt.lr_policy != "linear":
            raise ValueError("Joint experiments currently support the fixed linear LR protocol")
        if opt.isTrain and opt.joint_smoke and (opt.n_epochs + opt.n_epochs_decay > 2 or opt.max_dataset_size > 2):
            raise ValueError("joint_smoke is limited to <=2 epochs and <=2 samples per domain")
        super().__init__(opt)
        self.netG_A = JointContourGenerator(
            self.netG_A.backbone, opt.condition_mode,
            contour_width=opt.contour_width, embedding_dim=opt.contour_embedding_dim,
            epsilon=opt.contour_epsilon, residual_limit=opt.contour_residual_limit,
            edge_sigma=opt.edge_sigma, edge_low_threshold=opt.edge_low_threshold,
            edge_high_threshold=opt.edge_high_threshold, edge_soft_width=opt.edge_soft_width,
            soft_high_thresholds=opt.soft_high_thresholds, saliency_sigmas=opt.saliency_sigmas,
            saliency_inner_weight=opt.saliency_inner_weight, saliency_outer_weight=opt.saliency_outer_weight,
            saliency_transition_fraction=opt.saliency_transition_fraction,
            saliency_background_gain=opt.saliency_background_gain,
        )
        config_keys = (
            "joint_variant", "condition_mode", "fusion_mode", "netG", "ngf", "norm",
            "no_dropout", "input_nc", "output_nc", "direction", "contour_width",
            "contour_embedding_dim", "contour_epsilon", "contour_residual_limit",
            "edge_sigma", "edge_low_threshold", "edge_high_threshold", "edge_soft_width",
            "soft_high_thresholds", "saliency_sigmas", "saliency_inner_weight",
            "saliency_outer_weight", "saliency_transition_fraction", "saliency_background_gain",
            "lambda_contour_anchor", "lambda_contour_nce", "contour_positive_threshold",
            "contour_background_threshold", "contour_gradient_threshold", "contour_exclusion_radius",
            "contour_max_anchors", "contour_max_negatives", "contour_temperature",
            "contour_intensity_jitter", "contour_sampling_seed",
        )
        self.joint_config = {key: getattr(opt, key) for key in config_keys}
        signature = hashlib.sha256(json.dumps(self.joint_config, sort_keys=True).encode()).digest()
        self.expected_signature = list(signature)
        self.netG_A.register_buffer("joint_signature", torch.tensor(self.expected_signature, dtype=torch.uint8))
        self.netG_A.register_buffer("joint_updates", torch.zeros((), dtype=torch.int64))
        self.visual_names += ["contour_prior", "contour_refined", "contour_change"]
        self._at_epoch_end, self._inactive_steps = False, 0
        if self.isTrain:
            self.loss_names += ["contour_anchor", "contour_NCE", "contour_NCE_raw", "contour_active_anchors",
                                "contour_change_mean", "contour_fraction", "contour_head_grad", "contour_channel_grad"]
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers[0] = self.optimizer_G

    def setup(self, opt):
        if self.isTrain and not opt.continue_train and any(self.save_dir.glob("*_net_*.pth")):
            raise FileExistsError("Joint experiment already has weights; use a new name or continue_train")
        if self.isTrain and not opt.continue_train and opt.epoch_count != 1:
            raise ValueError("Fresh training must start at epoch_count=1")
        super().setup(opt)
        if self.netG_A.joint_signature.cpu().tolist() != self.expected_signature:
            raise ValueError("Checkpoint joint configuration differs; pass its original contour/variant options")
        if self.isTrain and not opt.continue_train:
            # BaseModel.setup initializes ALL convolutions, including the head.
            self.netG_A.refiner.reset_to_prior()
            initial = hashlib.sha256()
            for name in self.model_names:
                for key, parameter in getattr(self, "net" + name).named_parameters():
                    initial.update(f"{name}:{key}".encode())
                    initial.update(parameter.detach().cpu().numpy().tobytes())
            self.initial_parameters_sha256 = initial.hexdigest()
        if self.isTrain:
            from data.image_folder import make_dataset
            data_digest = hashlib.sha256()
            for split in ("trainA", "trainB"):
                directory = Path(opt.dataroot) / split
                files = sorted(make_dataset(str(directory), opt.max_dataset_size))
                if not files:
                    raise ValueError(f"No training images in {directory}")
                for filename in files:
                    path = Path(filename)
                    data_digest.update(f"{split}/{path.relative_to(directory).as_posix()}".encode())
                    data_digest.update(sha256_file(path).encode())
            self.training_data_sha256 = data_digest.hexdigest()
        if self.isTrain and opt.continue_train:
            suffix = f"iter_{opt.load_iter}" if opt.load_iter > 0 else opt.epoch
            self._restore_training(suffix)

    def forward(self):
        details = self.netG_A(self.real_A, return_details=True, condition_override=self.opt.contour_intervention)
        self.joint_details = details
        self.fake_B = details["rgb"]
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A, condition_override=self.opt.contour_intervention)
        # Existing visualizer expects [-1,1]; actual contour computations use [0,1].
        self.contour_prior = details["prior"].detach() * 2 - 1
        self.contour_refined = details["edge"].detach() * 2 - 1
        clipped_prior = details["prior"].clamp(self.opt.contour_epsilon, 1 - self.opt.contour_epsilon)
        self.contour_change = (details["edge"] - clipped_prior).abs().detach() * 2 - 1
        if self.isTrain:
            step = int(self.netG_A.joint_updates)
            view_rng = torch.Generator().manual_seed(self.opt.contour_sampling_seed + step * 2)
            code = int(torch.randint(8, (), generator=view_rng))
            jitter = self.opt.contour_intensity_jitter
            gain = 1 + (float(torch.rand((), generator=view_rng)) * 2 - 1) * jitter
            offset = (float(torch.rand((), generator=view_rng)) * 2 - 1) * jitter
            augmented = self.netG_A(
                transform_view((self.real_A * gain + offset).clamp(-1, 1), code), contour_only=True,
                prior_override=transform_view(details["prior"], code))
            self.aligned_augmented_edge = transform_view(augmented["edge"], code, inverse=True)

    def backward_G(self):
        # Same original CycleGAN losses and weights in both variants.
        self.loss_idt_A = self.loss_idt_B = 0
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        prior, edge = self.joint_details["prior"], self.joint_details["edge"]
        positive, negative = heuristic_anchors(
            self.real_A, prior, self.opt.contour_positive_threshold,
            self.opt.contour_background_threshold, self.opt.contour_gradient_threshold,
            self.opt.contour_exclusion_radius)
        self.loss_contour_anchor = self.opt.lambda_contour_anchor * (
            anchor_loss(edge, prior, positive, negative, self.opt.contour_epsilon)
            + anchor_loss(self.aligned_augmented_edge, prior, positive, negative, self.opt.contour_epsilon)) / 2
        raw_nce, active = edge.sum() * 0, 0
        if self.opt.joint_variant == "contrastive":
            rng = torch.Generator().manual_seed(self.opt.contour_sampling_seed + int(self.netG_A.joint_updates) * 2 + 1)
            raw_nce, active = local_map_contrast(
                self.netG_A.projection(edge), self.netG_A.projection(self.aligned_augmented_edge),
                positive, negative, rng, self.opt.contour_max_anchors, self.opt.contour_max_negatives,
                self.opt.contour_temperature, self.opt.contour_exclusion_radius)
            self._inactive_steps = 0 if active else self._inactive_steps + 1
            if self._inactive_steps == 100:
                warnings.warn("100 consecutive steps without contrastive anchors; inspect prior/anchor coverage")
        self.loss_contour_NCE_raw = raw_nce.detach()
        self.loss_contour_NCE = raw_nce * self.opt.lambda_contour_nce
        self.loss_contour_active_anchors = active
        self.loss_contour_change_mean = (edge.detach() - prior.clamp(self.opt.contour_epsilon, 1 - self.opt.contour_epsilon)).abs().mean()
        self.loss_contour_fraction = (edge.detach() >= 0.5).float().mean()
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
                       + self.loss_contour_anchor + self.loss_contour_NCE)
        if not torch.isfinite(self.loss_G):
            raise RuntimeError("Non-finite joint generator loss")
        self.loss_G.backward()
        head_grad = self.netG_A.refiner.residual_head.weight.grad
        self.loss_contour_head_grad = float(head_grad.norm()) if head_grad is not None else 0.0
        first_conv = next(module for module in self.netG_A.backbone.modules() if isinstance(module, torch.nn.Conv2d))
        self.loss_contour_channel_grad = float(first_conv.weight.grad[:, 1].norm()) if first_conv.weight.grad is not None else 0.0

    def optimize_parameters(self):
        self._at_epoch_end = False
        super().optimize_parameters()
        self.netG_A.joint_updates.add_(1)

    def update_learning_rate(self):
        super().update_learning_rate()
        self._at_epoch_end = True

    def _training_config(self):
        return {key: getattr(self.opt, key) for key in (
            "n_epochs", "n_epochs_decay", "lr", "beta1", "lambda_A", "lambda_B", "lambda_identity",
            "lr_policy", "pool_size", "seed", "batch_size", "preprocess", "crop_size", "load_size",
            "no_flip", "serial_batches", "max_dataset_size", "netD", "ndf", "n_layers_D", "gan_mode")}

    def save_networks(self, epoch):
        checksums = {}
        for name in self.model_names:
            path = self.save_dir / f"{epoch}_net_{name}.pth"
            atomic_save(getattr(self, "net" + name).state_dict(), path)
            checksums[name] = sha256_file(path)
        if self.isTrain:
            metadata = {"format": "joint_contour_v1", "joint_config": self.joint_config,
                        "training_config": self._training_config(), "training_data_sha256": self.training_data_sha256,
                        "initial_parameters_sha256": self.initial_parameters_sha256,
                        "scope": "implementation_smoke" if self.opt.joint_smoke else "quality_unverified",
                        "updates": int(self.netG_A.joint_updates), "at_epoch_end": self._at_epoch_end,
                        "checksums": checksums}
            metadata_path = self.save_dir / f"{epoch}_joint_metadata.json"
            temporary = metadata_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            temporary.replace(metadata_path)
            numpy_state = np.random.get_state()
            state = {"format": "joint_contour_training_v1", "joint_config": self.joint_config,
                     "training_config": self._training_config(), "checksums": checksums,
                     "training_data_sha256": self.training_data_sha256,
                     "initial_parameters_sha256": self.initial_parameters_sha256,
                     "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                     "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
                     "at_epoch_end": self._at_epoch_end,
                     "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                     "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "image_pools": [getattr(self, name).images if getattr(self, name).pool_size else []
                                     for name in ("fake_A_pool", "fake_B_pool")]}
            atomic_save(state, self.save_dir / f"{epoch}_joint_training.pt")

    def _restore_training(self, suffix):
        path = self.save_dir / f"{suffix}_joint_training.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "joint_contour_training_v1":
            raise ValueError("Unsupported joint training checkpoint")
        if json.dumps(state["joint_config"], sort_keys=True) != json.dumps(self.joint_config, sort_keys=True) or state["training_config"] != self._training_config():
            raise ValueError("Resume configuration differs from the original joint run")
        if state["training_data_sha256"] != self.training_data_sha256:
            raise ValueError("Resume training images differ from the original run")
        self.initial_parameters_sha256 = state["initial_parameters_sha256"]
        for name, expected in state["checksums"].items():
            if sha256_file(self.save_dir / f"{suffix}_net_{name}.pth") != expected:
                raise ValueError("Incomplete/mixed checkpoint: network checksum mismatch")
        if not state["at_epoch_end"]:
            raise ValueError("Strict budget resume requires an end-of-epoch checkpoint; choose a numbered epoch")
        completed = state["schedulers"][0]["last_epoch"]
        if self.opt.epoch_count != completed + 1:
            raise ValueError(f"Resume requires --epoch_count {completed + 1}")
        # Rebuild with the ORIGINAL epoch origin; avoid applying epoch_count twice.
        schedule_opt = copy.copy(self.opt)
        schedule_opt.epoch_count = 1
        self.schedulers = [networks.get_scheduler(optimizer, schedule_opt) for optimizer in self.optimizers]
        for optimizer, saved in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(self.schedulers, state["schedulers"]):
            scheduler.load_state_dict(saved)
        for name, images in zip(("fake_A_pool", "fake_B_pool"), state["image_pools"]):
            pool = getattr(self, name)
            if pool.pool_size:
                pool.images = [image.to(self.device) for image in images]
                pool.num_imgs = len(images)
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self._at_epoch_end = True
