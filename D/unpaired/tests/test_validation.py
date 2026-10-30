import copy
import csv
import io
import random
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from D.common import read_json, write_json, sha256
from D.unpaired import checkpoint
from D.unpaired.evaluate import checked_metrics, checked_prediction, evaluate_predictions
from D.unpaired.protocol import config_from, IndexStream
from D.unpaired.validation import (TrainingValidation, candidate_record, full_validation,
    rank, register_best, selected_record, validation_identity)
from D.unpaired.tests.test_stability import toy_engine


def metric_values(count=3):
    return dict(fid=142., kid_mean=.1, kid_std=.01, apce_py=.3, edge_precision_py=.25,
                edge_recall_py=.3, edge_f1_py=.27, num_tir_inputs=count, num_generated=count,
                num_reference_rgb=count, generated_suffix="_fake_B.png", kid_subset_size=50,
                kid_subsets=100, edge_canny_sigma=1., device="cpu", batch_size=16)


class ValidationFixture:
    def __init__(self, root, count=3):
        self.root, self.count = Path(root), count
        self.run = self.root / "run"; self.run.mkdir()
        self.cache = self.root / "conditions"; self.cache.mkdir()
        (self.cache / "maps").mkdir()
        self.data = self.root / "data.json"
        self.rows = dict(A=[], B=[])
        frames = []
        for domain in ("A", "B"):
            for index in range(count):
                name = f"{domain}{index:03d}"
                Image.new("RGB", (8,8), (index+30,70,110)).save(self.root / f"{name}.png")
                self.rows[domain].append(dict(image=f"{name}.png", split="val", group=f"{domain}_val"))
                if domain == "B":
                    rgb=np.asarray(Image.open(self.root / f"{name}.png").convert('RGB')).copy()
                    p=np.zeros((15,8,8),dtype=np.float32);p[1]=1
                    relative=f"maps/{name}.npz"
                    np.savez_compressed(self.cache / relative,RGB=rgb,gray=rgb[:,:,0],
                                        E=np.zeros((8,8),np.float32),P=p,Q=np.ones((8,8),np.float32))
                    frames.append(dict(id=f"TIR_val_{name}", filename=f"{name}.png", stem=name,
                                       domain="TIR", role="val", source=f"{name}.png", native_size=[8,8],
                                       source_sha256=sha256(self.root / f"{name}.png"),condition=relative,
                                       condition_sha256=sha256(self.cache / relative)))
        write_json(self.data,dict(format="grouped_unaligned_v1",domains=self.rows))
        self.identity = dict(seed=42, source_manifest_sha256=sha256(self.data), recipe={"fixture":1})
        write_json(self.cache / "manifest.json",dict(format="D_unpaired_conditions_v1",identity=self.identity,frames=frames))
        write_json(self.cache / "status.json",dict(phase="completed",completed=count,total=count))
        write_json(self.run / "run.json",dict(format="D_unpaired_run_v1",config={"seed":42},
                   data={"source":self.identity},prior_sha256="fixture"))
        self.engine = toy_engine()
        self.streams = dict(TIR=IndexStream(count,"TIR"),RGB=IndexStream(count,"RGB"))

    def prediction(self, step=1):
        location = checkpoint.save(self.run,step,self.engine,self.streams)
        output = self.root / f"step_{step}" / "predictions/adapted"
        (output / "images").mkdir(parents=True)
        frames = read_json(self.cache / "manifest.json")["frames"]
        records = {}
        for frame in frames:
            target = output / "images" / f"{frame['stem']}_fake_B.png"
            Image.new("RGB",(8,8),(80,90,100)).save(target)
            records[frame["stem"]] = dict(source=frame["source"], source_sha256=sha256(self.root / frame["source"]),
                                        sha256=sha256(target), raw_model_output=True)
        header = read_json(location / "checkpoint.json")
        write_json(output / "prediction.json",dict(format="D_unpaired_prediction_v1",frames=[r["id"] for r in frames],
                   condition_manifest_sha256=sha256(self.cache / "manifest.json"), metadata=dict(phase="val",step=step,
                   run_sha256=sha256(self.run / "run.json"),generator_sha256=header["files"]["generator.safetensors"])))
        write_json(output / "records.json",records)
        write_json(output / "status.json",dict(phase="completed",count=self.count))
        return output

    def fake_evaluator(self, arguments, **keywords):
        output = Path(arguments[arguments.index("--output-dir")+1]); output.mkdir(exist_ok=True)
        write_json(output / "metrics.json",metric_values(self.count))
        with (output / "evaluation_manifest.csv").open("w",newline="",encoding="utf8") as handle:
            writer=csv.writer(handle);writer.writerow(("tir_filename","generated_filename"))
            writer.writerows((f"B{i:03d}.png",f"B{i:03d}_fake_B.png") for i in range(self.count))
        for name in ("edge_threshold_metrics.csv","metric_report.png","metric_report.pdf"):
            (output / name).write_bytes(b"synthetic unit-test report")

    def evaluate(self, prediction, *, resume=False):
        directory=prediction.parent.parent / "evaluation"
        with patch("D.unpaired.evaluate.subprocess.run",side_effect=self.fake_evaluator):
            evaluate_predictions(prediction,self.cache,self.data,directory,device="cpu",resume=resume)
        return directory / "metrics/metrics.json"

    def record(self, step=1):
        prediction=self.prediction(step)
        metrics=self.evaluate(prediction)
        return candidate_record(self.run,step,prediction,metrics,self.cache,self.data)


class FullValidationTests(unittest.TestCase):
    def test_rejects_twelve_image_style_subset_even_with_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp)
            path=fixture.cache / "manifest.json";cache=read_json(path);cache["frames"].pop();write_json(path,cache)
            write_json(fixture.cache / "status.json",dict(phase="completed",completed=2,total=2))
            with self.assertRaisesRegex(ValueError,"complete validation"):
                full_validation(fixture.cache,fixture.data)

    def test_test_predictions_cannot_be_used_for_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);prediction=fixture.prediction()
            path=prediction / "prediction.json";info=read_json(path);info["metadata"]["phase"]="test";write_json(path,info)
            with self.assertRaisesRegex(ValueError,"never test"):
                checked_prediction(prediction,fixture.cache,fixture.data,required_phase="val")

    def test_changed_raw_rgb_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);prediction=fixture.prediction()
            Image.new("RGB",(8,8),"red").save(prediction / "images/B000_fake_B.png")
            with self.assertRaisesRegex(ValueError,"raw generated image changed"):
                checked_prediction(prediction,fixture.cache,fixture.data)

    def test_nonfinite_and_wrong_count_scores_fail_but_negative_unbiased_kid_is_allowed(self):
        for key,value in (("fid",float("nan")),("kid_mean",float("inf")),("edge_f1_py",1.1),("num_generated",2)):
            values=metric_values();values[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):checked_metrics(values,3,3)
        values=metric_values();values["kid_mean"]=-.001
        self.assertEqual(checked_metrics(values,3,3)["kid_mean"],-.001)


class EvaluationTransactionTests(unittest.TestCase):
    def test_completed_evaluation_resumes_without_rescoring_and_detects_report_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);prediction=fixture.prediction();metrics=fixture.evaluate(prediction)
            with patch("D.unpaired.evaluate.subprocess.run") as runner:
                evaluate_predictions(prediction,fixture.cache,fixture.data,metrics.parent.parent,device="cpu",resume=True)
                runner.assert_not_called()
            metrics.write_text('{}',encoding="utf8")
            with self.assertRaisesRegex(ValueError,"report changed"):
                evaluate_predictions(prediction,fixture.cache,fixture.data,metrics.parent.parent,device="cpu",resume=True)

    def test_failed_metrics_do_not_publish_best_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);prediction=fixture.prediction();directory=prediction.parent.parent / "evaluation"
            with patch("D.unpaired.evaluate.subprocess.run",side_effect=subprocess.CalledProcessError(1,["fixture"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    evaluate_predictions(prediction,fixture.cache,fixture.data,directory,device="cpu")
            self.assertEqual(read_json(directory / "evaluation_status.json")["phase"],"failed")
            self.assertFalse((fixture.run / "best.json").exists())
            fixture.evaluate(prediction,resume=True)
            self.assertEqual(read_json(directory / "evaluation_status.json")["phase"],"completed")

    def test_legacy_import_verifies_actual_exported_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);prediction=fixture.prediction();metrics=fixture.evaluate(prediction)
            directory=metrics.parent.parent
            (directory / "evaluation.json").unlink();(directory / "evaluation_status.json").unlink()
            result=candidate_record(fixture.run,1,prediction,metrics,fixture.cache,fixture.data,import_existing=True)
            self.assertIn("scores not recomputed",result["evidence"])
            Image.new("RGB",(8,8),"red").save(directory / "reference_RGB/A000.png")
            with self.assertRaisesRegex(ValueError,"input changed"):
                candidate_record(fixture.run,1,prediction,metrics,fixture.cache,fixture.data,import_existing=True)


class BestSelectionTests(unittest.TestCase):
    def test_structure_first_keeps_separate_distribution_winners_and_resolves_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);a=fixture.record(1);b=fixture.record(2)
            # Change only the synthetic score fixture; checkpoint binding remains real.
            a["metrics"].update(edge_f1_py=.30,fid=140.,kid_mean=.09)
            b["metrics"].update(edge_f1_py=.32,fid=155.,kid_mean=.12)
            directory=fixture.root / "selection"
            register_best(directory,a);state=register_best(directory,b)
            self.assertEqual(state["best"]["primary"],b["id"])
            self.assertEqual(state["best"]["fid"],a["id"])
            self.assertEqual(state["best"]["kid"],a["id"])
            self.assertEqual(selected_record(directory / "best.json")[2]["step"],2)
            self.assertEqual(selected_record(directory / "best.json","fid")[2]["step"],1)
            self.assertEqual(len(register_best(directory,b)["records"]),2)
            with self.assertRaisesRegex(ValueError,"preserve its policy"):
                register_best(directory,b,policy="kid")
            changed=copy.deepcopy(b);changed["metrics"]["fid"]=150.
            with self.assertRaisesRegex(ValueError,"evidence changed"):
                register_best(directory,changed)

    def test_ties_use_fid_then_kid_then_earliest_update(self):
        a=dict(step=1,run_sha256="same",metrics=metric_values())
        b=copy.deepcopy(a);b["step"]=2
        self.assertLess(rank(a,"structure"),rank(b,"structure"))
        b["metrics"]["fid"]-=1
        self.assertLess(rank(b,"structure"),rank(a,"structure"))
        b["metrics"]["fid"]=a["metrics"]["fid"];b["metrics"]["kid_mean"]-=.01
        self.assertLess(rank(b,"structure"),rank(a,"structure"))

    def test_metric_evidence_for_another_step_and_rolling_snapshots_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);prediction=fixture.prediction(1);metrics=fixture.evaluate(prediction)
            checkpoint.save(fixture.run,2,fixture.engine,fixture.streams)
            with self.assertRaisesRegex(ValueError,"do not belong"):
                candidate_record(fixture.run,2,prediction,metrics,fixture.cache,fixture.data)
            checkpoint.save(fixture.run,3,fixture.engine,fixture.streams,kind="recovery")
            with self.assertRaisesRegex(ValueError,"immutable archival"):
                candidate_record(fixture.run,3,prediction,metrics,fixture.cache,fixture.data)

    def test_failed_atomic_best_write_keeps_the_previous_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);a=fixture.record(1);b=fixture.record(2)
            directory=fixture.root / "selection";register_best(directory,a)
            before=(directory / "best.json").read_bytes()
            with patch("D.unpaired.validation.write_json",side_effect=OSError("disk full")):
                with self.assertRaises(OSError):register_best(directory,b)
            self.assertEqual((directory / "best.json").read_bytes(),before)

    def test_validation_cache_cannot_be_silently_enabled_with_a_different_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp)
            args=SimpleNamespace(validation_cache=fixture.cache,data_manifest=fixture.data,save_every=1)
            with self.assertRaisesRegex(ValueError,"training recipe"):
                validation_identity(args,{"different":True})


class ValidationIsolationTests(unittest.TestCase):
    def test_full_callback_generates_verified_images_and_registers_real_checkpoint_once(self):
        from D.unpaired.engine import UnpairedEngine
        class PreviewGenerator(nn.Module):
            def __init__(self):super().__init__();self.conv=nn.Conv2d(3,3,1)
            def forward(self,x,p,q):return self.conv(x).tanh()
            def export_state(self):return {k:v.detach().clone() for k,v in self.state_dict().items()}
            def import_state(self,state):self.load_state_dict(state)
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp,count=50)
            engine=UnpairedEngine(PreviewGenerator(),nn.Linear(1,1),nn.ModuleDict(dict(D=nn.Linear(1,1))),
                                  None,None,config_from(),device='cpu')
            checkpoint.save(fixture.run,1,engine,fixture.streams)
            args=SimpleNamespace(validation_cache=fixture.cache,data_manifest=fixture.data)
            identity=dict(num_tir=50,num_rgb=50,every=1,policy='structure',metric_environment={'device':'cpu'},
                          cache_manifest_sha256=sha256(fixture.cache / 'manifest.json'))
            runner=TrainingValidation(fixture.run,args,identity)
            before=checkpoint.capture_rng()
            with patch('D.unpaired.evaluate.subprocess.run',side_effect=fixture.fake_evaluator) as metrics_runner, \
                 redirect_stdout(io.StringIO()):
                runner.run(engine,1,SimpleNamespace(event=lambda *args:None))
                runner.run(engine,1,SimpleNamespace(event=lambda *args:None))
            self.assertEqual(metrics_runner.call_count,1)
            run,location,record=selected_record(fixture.run / 'best.json')
            self.assertEqual((run,location,record['step']),(fixture.run,fixture.run / 'checkpoints/step_000001',1))
            self.assertEqual(record['num_tir'],50)
            self.assertEqual(read_json(fixture.run / 'validation/step_000001/predictions/adapted/status.json')['count'],50)
            self.assertEqual(len(read_json(fixture.run / 'best.json')['records']),1)
            self.assertTrue(torch.equal(before['cpu_rng'],checkpoint.capture_rng()['cpu_rng']))

    def test_next_real_optimizer_update_matches_with_or_without_validation(self):
        from D.unpaired.engine import UnpairedEngine
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv=nn.Conv2d(3,3,1);self.dropout=nn.Dropout(.2);self.fusion=nn.Identity()
            def forward(self,x,p,q,*,generator=None):
                noise=torch.randn(x.shape,generator=generator or torch.Generator().manual_seed(42))*.01
                return (self.dropout(self.conv(x))+noise).tanh()
            def export_state(self):return {k:v.detach().clone() for k,v in self.state_dict().items()}
            def import_state(self,state):self.load_state_dict(state)
        def make_engine():
            config=config_from()
            config['loss_weights']={key:0. for key in config['loss_weights']}
            config['loss_weights'].update(adversarial=1.,cycle_tir=10.)
            return UnpairedEngine(Generator(),nn.Sequential(nn.Conv2d(3,1,1),nn.Tanh()),
                nn.ModuleDict(dict(RGB=nn.Conv2d(3,1,1),TIR=nn.Conv2d(1,1,1))),None,None,config,device='cpu')
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);engine=make_engine()
            sample=dict(x=torch.rand(1,3,8,8),p=torch.ones(1,15,8,8)/15,q=torch.ones(1,1,8,8),rgb=torch.rand(1,3,8,8)*2-1)
            self.assertTrue(engine.update(sample,sample,0,4)[0])
            saved=checkpoint.save(fixture.run,1,engine,fixture.streams)
            self.assertTrue(engine.update(sample,sample,1,4)[0])
            expected=[copy.deepcopy(m.state_dict()) for m in (engine.generator,engine.reverse,engine.critics)]
            restored=make_engine();streams=dict(TIR=IndexStream(3,'TIR'),RGB=IndexStream(3,'RGB'))
            checkpoint.restore(fixture.run,saved,restored,streams)
            runner=TrainingValidation.__new__(TrainingValidation)
            runner.folder,runner.dataset=fixture.run,object()
            runner.args=SimpleNamespace(validation_cache=fixture.cache,data_manifest=fixture.data)
            runner.identity=dict(num_tir=3,num_rgb=3,policy='structure',metric_environment={'device':'cpu'})
            record=dict(id='fixture',step=1,metrics=metric_values())
            state=dict(policy='structure',records=[record],best={'primary':'fixture'})
            def noisy_preview(*args,**kwargs):
                torch.randn(10);random.random();np.random.random();restored.generator.eval()
                torch.randn(10,generator=restored.noise_rng);torch.randn(10,generator=restored.patch_rng)
                restored.rgb_pool.query(torch.randn(1,3,8,8))
            with patch('D.unpaired.report.preview',side_effect=noisy_preview), \
                 patch('D.unpaired.validation.evaluate_predictions'), \
                 patch('D.unpaired.validation.candidate_record',return_value=record), \
                 patch('D.unpaired.validation.register_best',return_value=state):
                runner.run(restored,1,SimpleNamespace(event=lambda *args:None))
            self.assertTrue(restored.update(sample,sample,1,4)[0])
            for module,values in zip((restored.generator,restored.reverse,restored.critics),expected):
                self.assertTrue(all(torch.equal(value,values[key]) for key,value in module.state_dict().items()))

    def test_validation_restores_all_random_streams_modes_and_diagnostics_even_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp);checkpoint.save(fixture.run,1,fixture.engine,fixture.streams)
            engine=fixture.engine
            engine.generator.dropout=nn.Dropout(.5).eval()
            engine.generator.fusion=SimpleNamespace(diagnostics={"before":1},collect_diagnostics=True)
            engine.generator.output_diagnostics={"before":2}
            runner=TrainingValidation.__new__(TrainingValidation)
            runner.folder,runner.dataset=fixture.run,object()
            runner.args=SimpleNamespace(validation_cache=fixture.cache,data_manifest=fixture.data)
            runner.identity=dict(num_tir=3,num_rgb=3,policy="structure",metric_environment={"device":"cpu"})
            before=checkpoint.capture_rng();retry=engine.retry_state()
            log=SimpleNamespace(event=lambda *args:None)
            def bad_preview(*args,**kwargs):
                random.random();np.random.random();torch.randn(3)
                torch.randn(3,generator=engine.noise_rng);torch.randn(3,generator=engine.patch_rng)
                engine.generator.eval();engine.generator.dropout.train()
                engine.generator.fusion.diagnostics={"changed":1}
                engine.generator.output_diagnostics={"changed":2}
                raise RuntimeError("forced validation failure")
            with patch("D.unpaired.report.preview",side_effect=bad_preview):
                with self.assertRaisesRegex(RuntimeError,"forced validation"):
                    runner.run(engine,1,log)
            after=checkpoint.capture_rng()
            self.assertTrue(torch.equal(before["cpu_rng"],after["cpu_rng"]))
            self.assertEqual(before["python_rng"],after["python_rng"])
            self.assertTrue(torch.equal(before["numpy_rng"]["keys"],after["numpy_rng"]["keys"]))
            self.assertEqual(before["numpy_rng"]["position"],after["numpy_rng"]["position"])
            self.assertTrue(torch.equal(retry["noise_rng"],engine.noise_rng.get_state()))
            self.assertTrue(torch.equal(retry["patch_rng"],engine.patch_rng.get_state()))
            self.assertTrue(engine.generator.training);self.assertFalse(engine.generator.dropout.training)
            self.assertEqual(engine.generator.fusion.diagnostics,{"before":1})
            self.assertEqual(engine.generator.output_diagnostics,{"before":2})
            self.assertFalse((fixture.run / "best.json").exists())

    def test_checkpoint_before_validation_failure_is_resumable_before_another_update(self):
        from D.unpaired.train import run_updates
        with tempfile.TemporaryDirectory() as tmp:
            fixture=ValidationFixture(tmp)
            class Data:
                def __len__(self):return 3
                def __getitem__(self,index):return dict(x=torch.ones(1),frame={"filename":str(index)})
            args=SimpleNamespace(steps=2,verify_updates=0,log_every=1,save_every=2,recovery_every=100,preview_every=0)
            engine=fixture.engine
            def update(*args,**kwargs):return True,{"G_total":torch.tensor(1.)}
            engine.update=update
            validation=SimpleNamespace(due=lambda step,total:step==1,
                                       run=lambda *args:(_ for _ in ()).throw(RuntimeError("validation stopped")))
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,"validation stopped"):
                run_updates(args,fixture.run,engine,fixture.streams,Data(),Data(),0,validation=validation)
            saved,step=checkpoint.locate(fixture.run);self.assertEqual(step,1)
            self.assertEqual(saved.parent.name,"checkpoints")
            retry=SimpleNamespace(due=lambda step,total:step==1,run=lambda *args:None)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run_updates(args,fixture.run,engine,fixture.streams,Data(),Data(),1,validation=retry),2)


if __name__ == "__main__":
    unittest.main()
