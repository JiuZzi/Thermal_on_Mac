"""Same-setting pretrained/adapted panels: RGB gray proxy and fixed TIR."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import ROOT, DEFAULT_CACHE, DEFAULT_CONDITIONS, read_json, write_json, read_condition, sha256
from .prepare_rgb import DEFAULT_RGB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/D/training/D_rgb_softE_pilot_seed42")
    parser.add_argument("--step", type=int)
    parser.add_argument("--rgb-cache", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--tir-conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tir-limit", type=int, default=0, help="0 keeps all fixed validation frames")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = read_json(args.run_dir / "run.json")
    step = args.step if args.step is not None else read_json(args.run_dir / "latest.json")["step"]
    checkpoint = args.run_dir / "checkpoints" / f"step_{step:06d}"
    metadata = read_json(checkpoint / "checkpoint.json")
    if (metadata["run_sha256"] != sha256(args.run_dir / "run.json") or
        metadata["adapter_sha256"] != sha256(checkpoint / "adapter.safetensors") or metadata["model"] != run["model"]):
        parser.error("Adapter does not match the selected run/model")
    if run["rgb_manifest_sha256"] != sha256(args.rgb_cache / "manifest.json"):
        parser.error("RGB validation cache differs from training")
    from .train import read_rgb_cache
    _, rgb_samples = read_rgb_cache(args.rgb_cache)
    tir_source = read_json(args.tir_conditions / "manifest.json")
    if (tir_source["phase"] != "val" or read_json(args.tir_conditions / "status.json")["phase"] != "completed"):
        parser.error("Use the fixed validation TIR cache")
    cases = []
    for frame, rgb, edge in rgb_samples:
        if frame["role"] == "val":
            target = Image.fromarray(rgb)
            cases.append(dict(domain="RGB", stem=frame["stem"], initial=target.convert("L").convert("RGB"),
                              target=target, edge=edge))
    frames = tir_source["frames"][:args.tir_limit] if args.tir_limit else tir_source["frames"]
    for frame in frames:
        initial, edge = read_condition(args.tir_conditions, frame)
        cases.append(dict(domain="TIR", stem=frame["stem"], initial=initial, edge=edge))
    output = args.run_dir / "reviews" / f"step_{step:06d}"
    identity = dict(run_sha256=sha256(args.run_dir / "run.json"), adapter_sha256=metadata["adapter_sha256"],
                    tir_manifest_sha256=sha256(args.tir_conditions / "manifest.json"),
                    cases=[dict(domain=c["domain"], stem=c["stem"]) for c in cases], config=run["model"])
    if output.exists():
        if not args.resume or read_json(output / "review.json") != identity:
            parser.error("Preserve existing review; use matching --resume")
    else:
        output.mkdir(parents=True); write_json(output / "review.json", identity); write_json(output / "records.json", {})
    for label in ("before", "after", "panels"):
        (output / label).mkdir(exist_ok=True)
    records = read_json(output / "records.json")
    import torch
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    from .models import load_pipeline, configure_device
    from .adaptation import load_adapter
    from .preview import generate_one
    pipeline = configure_device(load_pipeline(run["model"], args.cache_dir), "cuda:0", True)
    pipeline.set_progress_bar_config(disable=True)
    for label in ("before", "after"):
        if label == "after":
            load_adapter(pipeline.controlnet, checkpoint / "adapter.safetensors")
        for case in cases:
            name = f"{case['domain']}_{case['stem']}.png"; key = f"{label}/{name}"
            path = output / key
            if key in records:
                if sha256(path) != records[key]["sha256"]:
                    raise ValueError("Review image changed")
                continue
            result, flagged = generate_one(pipeline, case["initial"], case["edge"], run["model"], "D1")
            result.resize(case["initial"].size, Image.Resampling.BICUBIC).save(path)
            records[key] = dict(sha256=sha256(path), safety_flagged=flagged)
            write_json(output / "records.json", records)
            print(f"{label} {case['domain']} {case['stem']}", flush=True)
    cards = []
    for index, case in enumerate(cases, 1):
        name = f"{case['domain']}_{case['stem']}.png"
        items = []
        if case["domain"] == "RGB":
            items.append(("RGB reference", case["target"]))
        items.extend([("TIR" if case["domain"] == "TIR" else "RGB gray proxy", case["initial"]),
                      ("clarity E", Image.fromarray(np.rint(case["edge"] * 255).astype(np.uint8)).convert("RGB"))])
        for label in ("before", "after"):
            with Image.open(output / label / name) as im:
                items.append(("pretrained" if label == "before" else f"adapted step {step}", im.convert("RGB")))
        w, h = case["initial"].size
        panel = Image.new("RGB", (w * len(items), h + 30), "white"); draw = ImageDraw.Draw(panel)
        for column, (caption, image) in enumerate(items):
            draw.text((column * w + 4, 7), caption, fill="black"); panel.paste(image, (column * w, 30))
        panel.save(output / "panels" / name)
        cards.append(f'<article><h3>{index:02d} {case["domain"]} {html.escape(case["stem"])}</h3>'
                     f'<a href="panels/{name}"><img loading="lazy" src="panels/{name}"></a></article>')
    page = '<!doctype html><meta charset="utf-8"><title>D training review</title>'
    page += '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}article{margin:24px 0}</style>'
    page += f'<h1>D RGB 自配对短程训练：step {step}</h1>'
    page += '<p>同一 seed42、尺寸、初始图、E、提示词和采样设置。仅比较条件分支训练前后。RGB 灰度是代理输入，TIR 没有配对 RGB 真值。</p>'
    page += '<p>先检查位置、数量、轮廓和凭空新增/删除的物体，再检查色彩。RGB 验证集改善而 TIR 没改善，说明跨域适配仍未解决；噪声损失下降不等于任务成功。</p>'
    page += f'<p>本轮模型工作尺寸 {run["model"]["width"]}×{run["model"]["height"]}、strength={run["model"]["strength"]}；不能与旧版640×512/0.6直接归因比较。</p>'
    (output / "index.html").write_text(page + ''.join(cards), encoding="utf-8")
    write_json(output / "status.json", dict(phase="completed", outputs=len(records), cases=len(cases)))
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
