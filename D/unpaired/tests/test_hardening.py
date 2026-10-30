"""Integration regressions for numerical updates, recovery and readable logging."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from D.common import ROOT, read_json, sha256, write_json
from D.unpaired import checkpoint
from D.unpaired.engine import NumericalFailure
from D.unpaired.engine import UnpairedEngine
from D.unpaired.losses import chroma_smoothness
from D.unpaired.progress import TrainingTimer
from D.unpaired.report import to_image, preview
from D.unpaired.train import run_updates, validate_recovery, source_hashes, RECOVERY_SOURCES
from D.unpaired.tests import test_stability as stability
from D.unpaired.tests.test_stability import FakeScaler, toy_engine


class EngineHardeningTests(unittest.TestCase):
    setUp = stability.AtomicUpdateTests.setUp
    tearDown = stability.AtomicUpdateTests.tearDown
    make_engine = stability.AtomicUpdateTests.make_engine

    def test_quality_policy_is_used_by_the_actual_training_engine(self):
        engine, sample = self.make_engine()
        engine.config["chroma_smooth_policy"] = "quality_soft"
        sample["q"] = sample["q"] * 0
        with patch("D.unpaired.engine.chroma_smoothness", wraps=chroma_smoothness) as smooth:
            applied, values = engine.update(sample, sample, 0, 10, diagnostics=True)
        self.assertTrue(applied)
        self.assertIs(smooth.call_args.args[3], sample["q"])
        self.assertEqual(float(values["chroma_smooth"]), 0.)
        weighted = sum(v for k, v in values.items() if k.startswith("weighted_"))
        torch.testing.assert_close(values["G_total"], weighted)
        self.assertIn("grad_norm_diffusion", values)
        self.assertIn("control_residual_rms", values)
        self.assertEqual(values["rgb_replay_active"], 1)

    def test_legacy_policy_keeps_ungated_smoothing(self):
        engine, sample = self.make_engine()
        sample["q"] = sample["q"] * 0
        with patch("D.unpaired.engine.chroma_smoothness", wraps=chroma_smoothness) as smooth:
            applied, _ = engine.update(sample, sample, 0, 10)
        self.assertTrue(applied)
        self.assertIsNone(smooth.call_args.args[3])

    def test_color_loss_black_white_and_unsupported_pixels_have_finite_gradients(self):
        _, sample = self.make_engine()
        for value in (-1., 0., 1.):
            for quality in (sample["q"], sample["q"] * 0):
                with self.subTest(value=value, unknown=not bool(quality.any())):
                    rgb = torch.full((1, 3, 64, 64), value, requires_grad=True)
                    loss, _ = self.model.prior.chroma_loss(rgb, sample["p"], quality)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss) and torch.isfinite(rgb.grad).all())
                    if not quality.any():
                        self.assertEqual(float(loss.detach()), 0.)
                        self.assertEqual(int(torch.count_nonzero(rgb.grad)), 0)

    def test_rgb_warmup_cannot_save_nonfinite_adapter_weights(self):
        from D.adaptation import save_adapter
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "adapter.safetensors"
            with torch.no_grad():
                self.model.controlnet.controlnet_cond_embedding.conv_in.weight[0, 0, 0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "nonfinite ControlNet"):
                save_adapter(self.model.controlnet, target)
            self.assertFalse(target.exists())

    def test_rejection_restores_global_rng_as_well_as_explicit_noise(self):
        engine, sample = self.make_engine()
        draws = []
        original = engine.generator.forward
        def stochastic(*args, **kwargs):
            draws.append(float(torch.rand(())))
            return original(*args, **kwargs)
        before = checkpoint.capture_rng()
        with patch.object(engine.generator, "forward", side_effect=stochastic):
            with patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("inf"))):
                applied, _ = engine.update(sample, sample, 0, 10)
            self.assertFalse(applied)
            self.assertTrue(torch.equal(torch.get_rng_state(), before["cpu_rng"]))
            count = len(draws)
            self.assertTrue(engine.update(sample, sample, 0, 10)[0])
            self.assertEqual(draws[:count], draws[count:])

    def test_commit_exception_is_not_described_as_a_safe_retry(self):
        engine, sample = self.make_engine()
        with patch.object(engine.opt_d, "step", side_effect=FloatingPointError("commit failure")):
            with self.assertRaisesRegex(FloatingPointError, "commit failure"):
                engine.update(sample, sample, 0, 10)
        self.assertEqual(engine.phase, "commit")
        self.assertEqual(engine.numerical_failures_total, 0)

    def test_weighted_loss_overflow_is_rejected_before_backward_or_commit(self):
        engine, sample = self.make_engine()
        engine.config["loss_weights"]["content"] = torch.finfo(torch.float32).max
        applied, report = engine.update(sample, sample, 0, 10)
        self.assertFalse(applied)
        self.assertEqual(report["stage"], "tir_losses")
        self.assertFalse(engine.opt_g.state)
        self.assertTrue(all(p.grad is None for p in engine.params_g))

    def test_full_optimizer_pool_and_rng_roundtrip_matches_uninterrupted_update(self):
        from D.unpaired.protocol import IndexStream
        engine, sample = self.make_engine()
        streams = dict(TIR=IndexStream(3, "TIR"), RGB=IndexStream(4, "RGB"))
        streams["TIR"].next(); streams["RGB"].next()
        self.assertTrue(engine.update(sample, sample, 0, 10)[0])
        self.assertTrue(engine.opt_g.state and engine.opt_d.state)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); write_json(folder / "run.json", {"fixture": 1})
            saved = checkpoint.save(folder, 1, engine, streams)
            expected_indices = {k: v.next() for k, v in streams.items()}
            self.assertTrue(engine.update(sample, sample, 1, 10)[0])
            expected = [copy.deepcopy(module.state_dict()) for module in (engine.generator, engine.reverse, engine.critics)]
            restored = UnpairedEngine(copy.deepcopy(engine.generator), copy.deepcopy(engine.reverse),
                                      copy.deepcopy(engine.critics), engine.builder, engine.rgb_teacher,
                                      engine.config, device="cpu")
            samplers = dict(TIR=IndexStream(3, "TIR"), RGB=IndexStream(4, "RGB"))
            self.assertEqual(checkpoint.restore(folder, saved, restored, samplers), 1)
            self.assertEqual({k: v.next() for k, v in samplers.items()}, expected_indices)
            self.assertTrue(restored.opt_g.state and restored.opt_d.state)
            self.assertTrue(restored.update(sample, sample, 1, 10)[0])
            for module, state in zip((restored.generator, restored.reverse, restored.critics), expected):
                self.assertTrue(all(torch.equal(value, state[key]) for key, value in module.state_dict().items()))

    def test_retry_limit_retains_unchanged_weights_and_positive_subunit_scale(self):
        engine, sample = self.make_engine()
        engine.scaler = FakeScaler()
        before = {k: v.clone() for k, v in engine.generator.state_dict().items()}
        with patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("inf"))):
            for _ in range(8):
                self.assertFalse(engine.update(sample, sample, 0, 10)[0])
            with self.assertRaises(NumericalFailure):
                engine.update(sample, sample, 0, 10)
        self.assertTrue(0 < engine.scaler.get_scale() < 1)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in engine.generator.state_dict().items()))


class CheckpointHardeningTests(unittest.TestCase):
    def test_cpu_only_rng_capture_does_not_initialize_an_available_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_initialized", return_value=False), \
             patch("torch.cuda.get_rng_state") as get_cuda_rng:
            state = checkpoint.capture_rng()
        self.assertIsNone(state["cuda_rng"])
        get_cuda_rng.assert_not_called()

    def fixture(self, folder):
        write_json(folder / "run.json", {"fixture": 1})
        from D.unpaired.protocol import IndexStream
        return toy_engine(), dict(TIR=IndexStream(3, "TIR"), RGB=IndexStream(3, "RGB"))

    def refresh_header(self, location, name):
        header = read_json(location / "checkpoint.json")
        header["files"][name] = sha256(location / name)
        write_json(location / "checkpoint.json", header)

    def test_optimizer_corruption_is_rejected_before_network_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams = self.fixture(folder)
            saved = checkpoint.save(folder, 0, engine, streams)
            state = torch.load(saved / "training.pt", weights_only=True)
            state["engine"]["opt_g"]["state"] = {0: {"exp_avg": torch.tensor(float("nan"))}}
            torch.save(state, saved / "training.pt"); self.refresh_header(saved, "training.pt")
            with torch.no_grad(): engine.generator.weight.add_(2)
            before = engine.generator.weight.clone()
            with self.assertRaisesRegex(ValueError, "optimizer history"):
                checkpoint.restore(folder, saved, engine, streams)
            torch.testing.assert_close(engine.generator.weight, before, rtol=0, atol=0)

    def test_nonfinite_reverse_checkpoint_cannot_partially_restore_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams = self.fixture(folder)
            saved = checkpoint.save(folder, 0, engine, streams)
            # Windows cannot overwrite a file while safetensors still maps it.
            state = {key:value.clone() for key,value in load_file(str(saved / "reverse.safetensors")).items()}
            state["weight"].fill_(float("inf"))
            save_file(state, str(saved / "reverse.safetensors")); self.refresh_header(saved, "reverse.safetensors")
            with torch.no_grad(): engine.generator.weight.add_(2)
            before = engine.generator.weight.clone()
            with self.assertRaisesRegex(ValueError, "nonfinite model"):
                checkpoint.restore(folder, saved, engine, streams)
            torch.testing.assert_close(engine.generator.weight, before, rtol=0, atol=0)

    def test_disk_error_keeps_latest_and_removes_only_owned_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams = self.fixture(folder)
            saved = checkpoint.save(folder, 0, engine, streams)
            before = (folder / "latest.json").read_bytes()
            with patch("D.unpaired.checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    checkpoint.save(folder, 1, engine, streams, kind="recovery")
            self.assertEqual((folder / "latest.json").read_bytes(), before)
            self.assertTrue(saved.is_dir())
            self.assertEqual(list((folder / "recovery").iterdir()), [])

    def test_commit_and_invalid_amp_states_are_not_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams = self.fixture(folder)
            engine.phase = "commit"
            with self.assertRaisesRegex(ValueError, "optimizer commit"):
                checkpoint.save(folder, 0, engine, streams)
            engine.phase = "ready"
            state = engine.state(); state["scaler"] = {"scale": float("nan")}
            with patch.object(engine, "state", return_value=state):
                with self.assertRaisesRegex(ValueError, "AMP scale"):
                    checkpoint.save(folder, 0, engine, streams)
            self.assertFalse((folder / "latest.json").exists())


class LoopAndLoggingTests(unittest.TestCase):
    def fixture(self, folder, total=2):
        write_json(folder / "run.json", {"fixture": 1})
        from D.unpaired.protocol import IndexStream
        engine = toy_engine()
        streams = dict(TIR=IndexStream(3, "TIR"), RGB=IndexStream(3, "RGB"))
        class Data:
            def __len__(self): return 3
            def __getitem__(self, index): return dict(frame=dict(filename=f"frame{index}"), x=torch.ones(1))
        args = SimpleNamespace(steps=total, verify_updates=0, log_every=10, save_every=total,
                               recovery_every=100, preview_every=0)
        return engine, streams, Data(), args

    def test_main_loop_uses_readable_log_and_retries_identical_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams, data, args = self.fixture(folder)
            attempts = []
            def update(tir, rgb, step, total, diagnostics=False):
                attempts.append((tir["frame"]["filename"], rgb["frame"]["filename"]))
                if len(attempts) == 1:
                    engine.phase = "rejected"
                    return False, dict(stage="fixture", amp_overflow=1, amp_scale_before=1., amp_scale_after=.5)
                engine.phase = "committed"
                return True, dict(G_total=torch.tensor(1.), edge=torch.tensor(.2))
            engine.update = update
            self.assertEqual(run_updates(args, folder, engine, streams, data, data, 0), 2)
            self.assertEqual(attempts[0], attempts[1])
            text = (folder / "train.log").read_text()
            for marker in ("[TRAIN", "[LOSS", "[AUX", "[RETRY", "[SAVED", "[DONE"):
                self.assertIn(marker, text)
            rows = [json.loads(line) for line in (folder / "loss.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in rows], [1, 2])
            self.assertEqual(rows[-1]["session_updates"], 2)
            self.assertEqual(rows[-1]["total_steps"], 2)
            self.assertEqual(read_json(folder / "latest.json")["step"], 2)

    def test_interrupt_after_commit_never_saves_the_old_counter_with_new_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams, data, args = self.fixture(folder)
            checkpoint.save(folder, 0, engine, streams)
            before = (folder / "latest.json").read_bytes()
            def update(*arguments, **keywords):
                with torch.no_grad(): engine.generator.weight.add_(.1)
                engine.phase = "committed"
                raise KeyboardInterrupt()
            engine.update = update
            with self.assertRaises(KeyboardInterrupt):
                run_updates(args, folder, engine, streams, data, data, 0)
            self.assertEqual((folder / "latest.json").read_bytes(), before)
            self.assertFalse((folder / "emergency").exists())

    def test_emergency_disk_error_does_not_mask_the_numerical_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine, streams, data, args = self.fixture(folder)
            def update(*arguments, **keywords):
                engine.phase = "rejected"
                raise NumericalFailure(dict(stage="fixture"))
            engine.update = update
            with patch("D.unpaired.train.checkpoints.save", side_effect=OSError("disk full")):
                with self.assertRaises(NumericalFailure):
                    run_updates(args, folder, engine, streams, data, data, 0)
            self.assertIn("disk full", read_json(folder / "failure.json")["recovery_save_error"])

    def test_eta_counts_this_resume_session_only(self):
        clock = iter((100., 106.))
        row = TrainingTimer(100, 200, clock=clock.__next__).snapshot(103)
        self.assertEqual(row["average_update_seconds"], 2.)
        self.assertEqual(row["eta_seconds"], 194.)

    def test_bad_preview_cannot_be_saved_as_a_plausible_black_image(self):
        with self.assertRaises(FloatingPointError):
            to_image(torch.full((1, 3, 8, 8), float("nan")))
        model = torch.nn.Linear(1, 1).train()
        with self.assertRaises(ValueError):
            preview(model, None, Path("unused"), limit=-1)
        self.assertTrue(model.training)


class AuditedMigrationTests(unittest.TestCase):
    def identities(self, revision):
        from D.unpaired.protocol import config_from
        manifests = read_json(ROOT / RECOVERY_SOURCES)["revisions"]
        files = manifests[revision]["files"]
        old = {name: values["crlf" if index % 2 else "lf"] for index, (name, values) in enumerate(files.items())}
        parent = dict(format="D_unpaired_run_v1", config=config_from(), data={"same": 1},
                      prior_sha256="same", versions={"torch": "same"}, initialization={"kind": "same"},
                      schedule=dict(steps=27655, save_every=5531, log_every=10, preview_every=5531), code_sha256=old)
        current = copy.deepcopy(parent); current["code_sha256"] = source_hashes()
        return parent, current

    def test_audited_revisions_accept_mixed_windows_line_endings_without_reset(self):
        for revision in ("6aec9bc3", "855396cf", "f2e53980"):
            with self.subTest(revision=revision):
                validate_recovery(*self.identities(revision))

    def test_unknown_source_and_changed_v2_objective_still_require_new_experiment(self):
        parent, current = self.identities("855396cf")
        parent["code_sha256"]["D/common.py"] = "f" * 64
        with self.assertRaises(ValueError): validate_recovery(parent, current)
        parent, current = self.identities("855396cf")
        current["code_sha256"]["D/common.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Unaudited"):
            validate_recovery(parent, current)
        parent, current = self.identities("f2e53980")
        for value in (parent, current):
            value["config"].update(format="D_unpaired_v2", chroma_smooth_policy="quality_soft")
        with self.assertRaisesRegex(ValueError, "quality_soft"):
            validate_recovery(parent, current)


class MetricInputTests(unittest.TestCase):
    def test_duplicate_reference_basenames_cannot_silently_reduce_the_rgb_set(self):
        from D.metric_inputs import rgb_reference_rows
        manifest = dict(domains=dict(A=[dict(image="camera1/frame.jpg", split="test"),
                                        dict(image="camera2/frame.png", split="test")]))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            rgb_reference_rows(manifest, "test")
        manifest["domains"]["A"][1]["image"] = "camera2/other.png"
        self.assertEqual(len(rgb_reference_rows(manifest, "test")), 2)


if __name__ == "__main__":
    unittest.main()
