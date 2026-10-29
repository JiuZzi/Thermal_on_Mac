"""Native-coordinate previews with explicit input, E, semantic P and baseline."""

import html
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from D.common import ROOT, read_json, write_json, sha256
from .data import on_device
from .progress import ItemProgress, duration, event


def to_image(values):
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] != 3:
        raise ValueError("Preview expects one RGB image [1,3,H,W]")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Refuse to save a nonfinite preview as a black/clamped image")
    pixels = ((values.detach().float().cpu()[0].clamp(-1,1) + 1) * 127.5).round().byte().permute(1,2,0).numpy()
    return Image.fromarray(pixels)


@torch.no_grad()
def preview(model, dataset, output, *, baseline=None, resume=False, limit=0, metadata=None, log=None):
    was_training = model.training
    try:
        model.eval()
        return _preview(model, dataset, output, baseline=baseline, resume=resume,
                        limit=limit, metadata=metadata, log=log)
    finally:
        model.train(was_training)


def _preview(model, dataset, output, *, baseline=None, resume=False, limit=0, metadata=None, log=None):
    if type(limit) is not int or limit < 0:
        raise ValueError("Preview limit must be nonnegative")
    output = Path(output)
    identity = dict(format="D_unpaired_prediction_v1", metadata=metadata,
                    condition_manifest_sha256=sha256(dataset.folder / "manifest.json"),
                    frames=[r["id"] for r in (dataset.frames[:limit] if limit else dataset.frames)])
    if output.exists():
        if not resume or read_json(output / "prediction.json") != identity:
            raise ValueError("Preserve existing preview; use matching --resume")
    else:
        output.mkdir(parents=True)
        for sub in ("images", "panels", "conditions"):
            (output / sub).mkdir()
        write_json(output / "prediction.json", identity); write_json(output / "records.json", {})
    records = read_json(output / "records.json")
    device = next(model.parameters()).device
    # Palette is display-only, never fed into the generator.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from semantic_preview.inference import PALETTE
    cards = []
    length = min(limit, len(dataset)) if limit else len(dataset)
    emit = log.event if log is not None else event
    emit("PREVIEW", f"{length} images | output: {output}")
    progress = ItemProgress(length, "PREVIEW", emit=emit)
    started = time.perf_counter()
    for index in range(length):
        sample = on_device(dataset[index], device)
        frame, name = sample["frame"], sample["frame"]["stem"]
        target = output / "images" / f"{name}_fake_B.png"
        if name in records:
            if sha256(target) != records[name]["sha256"]:
                raise ValueError("Preview output changed")
            with Image.open(target) as cached:
                current = cached.convert("RGB")
        else:
            values = model(sample["x"], sample["p"], sample["q"])
            current = to_image(values).resize(tuple(frame["native_size"]), Image.Resampling.BICUBIC)
            current.save(target)
            records[name] = dict(sha256=sha256(target), source=frame["source"],
                                 source_sha256=frame["source_sha256"], raw_model_output=True)
            write_json(output / "records.json", records)
        gray = (sample["x"][0,0].cpu().numpy() * 255).round().astype(np.uint8)
        edge = (sample["x"][0,1].cpu().numpy() * 255).round().astype(np.uint8)
        semantic = PALETTE[sample["p"].argmax(1)[0].cpu().numpy()]
        quality = (sample["q"][0,0].cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(quality).save(output / "conditions" / f"{name}_Q_semantic.png")
        items = [("TIR", Image.fromarray(gray).convert("RGB")), ("clarity E", Image.fromarray(edge).convert("RGB")),
                 ("semantic P", Image.fromarray(semantic))]
        if baseline is not None:
            path = Path(baseline) / "images" / f"{name}_fake_B.png"
            with Image.open(path) as before:
                items.append(("before adaptation", before.convert("RGB").resize((gray.shape[1],gray.shape[0]))))
        items.append(("generated RGB", current.resize((gray.shape[1],gray.shape[0]))))
        w, h = gray.shape[1], gray.shape[0]
        panel = Image.new("RGB", (len(items)*w,h+30), "white"); draw = ImageDraw.Draw(panel)
        for column,(caption,image) in enumerate(items):
            draw.text((column*w+4,7), caption, fill="black"); panel.paste(image,(column*w,30))
        panel_name = f"{name}.png"; panel.save(output / "panels" / panel_name)
        cards.append(f'<article><h3>{index+1:02d} {html.escape(frame["filename"])}</h3>'
                     f'<a href="panels/{panel_name}"><img loading="lazy" src="panels/{panel_name}"></a></article>')
        progress.update(index+1)
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>D TIR unpaired</title>'
    page += '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}article{margin:24px 0}</style>'
    page += '<h1>D：真实TIR无配对适配</h1><p>输入固定为[TIR,清晰E,0]。语义P与颜色统计是同源辅助条件。每张预览seed42，第三通道尚未训练。</p>'
    page += '<p>先看人车位置、数量、肢体、道路与建筑，再看颜色和割裂；场景错位则未达标。Q图是语义可靠性，不是轮廓置信度。没有逐帧配对RGB真值。</p>'
    page += '<p>这些是原始模型输出，不是经滤镜修饰的图像。短程示例不构成完整测试集的FID/KID结论。</p>'
    (output / "index.html").write_text(page + ''.join(cards) + '</html>', encoding="utf-8")
    write_json(output / "status.json", dict(phase="completed", count=length))
    emit("PREVIEW", f"done in {duration(time.perf_counter()-started)} | {output / 'index.html'}")
    return output / "index.html"
