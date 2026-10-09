"""Checks for alignment, unknown labels, fair sampling and actual gradients."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pytorch-CycleGAN-and-pix2pix"))

from contour_learning.data import ContourDataset, read_annotations
from contour_learning.model import SoftContourNet, transform_view
from contour_learning.objectives import boundary_loss, local_contrastive_loss
from models.conditioned_generator import ConditionedGenerator
from smoke_learned_contours import make_synthetic_data
from train_learned_contours import parse_args, run


class LearnedContourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_geometric_transforms_are_exactly_invertible(self):
        coordinate_map = torch.arange(3 * 5).reshape(1, 1, 3, 5)
        for code in range(8):
            restored = transform_view(transform_view(coordinate_map, code), code, inverse=True)
            torch.testing.assert_close(restored, coordinate_map, rtol=0, atol=0)

    def test_unknown_pixels_have_zero_task_gradient(self):
        labels = torch.tensor([[[[0, 255, 128]]]], dtype=torch.uint8)
        logits = torch.zeros_like(labels, dtype=torch.float32, requires_grad=True)
        loss = boundary_loss(logits, labels)
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 0, 2].item(), 0)
        self.assertGreater(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertLess(logits.grad[0, 0, 0, 1].item(), 0)
        with self.assertRaises(ValueError):
            boundary_loss(logits, torch.full_like(labels, 128))

    def test_local_loss_prefers_aligned_edges_over_wrong_features(self):
        labels = torch.tensor([[[[255, 0, 128, 0]]]], dtype=torch.uint8)
        a = torch.tensor([[[[1., -1., 0., -1.]], [[0., 0., 1., 0.]]]], requires_grad=True)
        good, active = local_contrastive_loss(a, a.clone(), labels, torch.Generator().manual_seed(1), exclusion_radius=0)
        wrong = a.detach().clone()
        wrong[:, :, :, 0] *= -1
        bad, _ = local_contrastive_loss(a, wrong, labels, torch.Generator().manual_seed(1), exclusion_radius=0)
        self.assertEqual(active, 1)
        self.assertLess(good.item(), bad.item())

    def test_cl_updates_encoder_projection_and_ignores_unknown_embeddings(self):
        torch.manual_seed(31)
        model = SoftContourNet(8, 8)
        image = torch.randn(1, 1, 16, 16)
        _, features_a = model(image)
        _, features_b = model(image * 0.9)
        a, b = model.projection(features_a), model.projection(features_b)
        a.retain_grad()
        labels = torch.zeros(1, 1, 16, 16, dtype=torch.uint8)
        labels[:, :, 8, 8] = 255
        labels[:, :, :4] = 128
        loss, active = local_contrastive_loss(a, b, labels, torch.Generator().manual_seed(3))
        loss.backward()
        self.assertGreater(active, 0)
        self.assertGreater(model.enc1[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.projection[2].weight.grad.abs().sum().item(), 0)
        self.assertEqual(a.grad[:, :, :4].abs().sum().item(), 0)

    def test_no_negatives_is_explicitly_inactive(self):
        a = torch.randn(1, 4, 4, 4, requires_grad=True)
        loss, active = local_contrastive_loss(a, a, torch.full((1, 1, 4, 4), 255), torch.Generator())
        self.assertEqual(active, 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_soft_edge_override_keeps_separate_channel_and_backbone_gradient(self):
        backbone = torch.nn.Conv2d(3, 1, 1, bias=False)
        generator = ConditionedGenerator(backbone, "zero", fusion_mode="direct")
        tir = torch.zeros(1, 1, 9, 11)
        scores = torch.rand_like(tir)
        generator(tir, edge_override=scores).sum().backward()
        self.assertGreater(backbone.weight.grad[:, 1].abs().sum().item(), 0)
        inspect = ConditionedGenerator(torch.nn.Identity(), "zero", fusion_mode="direct")
        actual = inspect(tir, edge_override=scores)
        torch.testing.assert_close(actual[:, 1:2], scores)
        self.assertEqual(actual[:, 2:].count_nonzero().item(), 0)

    def test_dataset_rejects_video_leakage_and_preserves_crop_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotations, _ = make_synthetic_data(Path(temporary) / "data")
            records, _ = read_annotations(annotations)
            a = ContourDataset(records, "train", crop_size=16, seed=31, edge_crop_probability=1)
            b = ContourDataset(records, "train", crop_size=16, seed=31, edge_crop_probability=1)
            a.epoch = b.epoch = 3
            for index in range(len(a)):
                image, label = a[index]
                other_image, other_label = b[index]
                torch.testing.assert_close(image, other_image)
                torch.testing.assert_close(label, other_label)
                self.assertTrue((label == 255).any())
                # Toy boundary always sits on the bright object's inner boundary.
                self.assertGreater(image[label == 255].min().item(), 0.2)
            with annotations.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[-1]["video_id"] = rows[0]["video_id"]
            with annotations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "video overlap"):
                read_annotations(annotations)

    def test_flir_protocol_rejects_test_images_and_content_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            annotations, _ = make_synthetic_data(folder / "data")
            with annotations.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            processed = folder / "processed"
            processed.mkdir()
            for row in rows:
                (processed / row["image"]).write_bytes((annotations.parent / row["image"]).read_bytes())
            protocol = folder / "protocol.csv"
            protocol_rows = [{"destination_split": "trainB", "destination_name": r["image"],
                              "video_id": r["video_id"]} for r in rows]
            def write_protocol():
                with protocol.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=protocol_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(protocol_rows)
            write_protocol()
            read_annotations(annotations, protocol, processed)
            protocol_rows[-1]["destination_split"] = "testB"
            write_protocol()
            with self.assertRaisesRegex(ValueError, "not in processed FLIR trainB"):
                read_annotations(annotations, protocol, processed)
            protocol_rows[-1]["destination_split"] = "trainB"
            write_protocol()
            with Image.open(processed / rows[0]["image"]) as image:
                array = np.array(image)
            array[0, 0] ^= 1
            Image.fromarray(array).save(processed / rows[0]["image"])
            with self.assertRaisesRegex(ValueError, "differs from processed trainB"):
                read_annotations(annotations, protocol, processed)

    def test_invalid_or_misaligned_label_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotations, _ = make_synthetic_data(Path(temporary) / "data")
            edge = annotations.parent / "toy_0_edge.png"
            with Image.open(edge) as image:
                labels = np.array(image)
            labels[10, 10] = 127
            Image.fromarray(labels).save(edge)
            with self.assertRaisesRegex(ValueError, "Invalid label values"):
                read_annotations(annotations)
            Image.fromarray(labels[:, :-1]).save(edge)
            with self.assertRaisesRegex(ValueError, "identical geometry"):
                read_annotations(annotations)

    def test_epoch_boundary_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            annotations, _ = make_synthetic_data(folder / "data")
            argv = ["--annotations", str(annotations), "--variant", "contrastive",
                    "--output-dir", str(folder / "training"), "--synthetic-smoke",
                    "--n-epochs", "1", "--n-epochs-decay", "1", "--save-every", "1",
                    "--crop-size", "32", "--width", "8", "--embedding-dim", "8", "--device", "cpu"]
            run(parse_args(argv))
            uninterrupted = torch.load(folder / "training/latest.pt", weights_only=True)
            run(parse_args(argv + ["--resume", str(folder / "training/epoch_001.pt")]))
            resumed = torch.load(folder / "training/latest.pt", weights_only=True)
            self.assertEqual(uninterrupted["history"], resumed["history"])
            for key in uninterrupted["model"]:
                torch.testing.assert_close(uninterrupted["model"][key], resumed["model"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
