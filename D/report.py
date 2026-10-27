"""Portable TIR / soft E / D0 / D1 contact sheets, without cherry-picking."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import DEFAULT_CONDITIONS, DEFAULT_OUTPUT, read_condition, read_json, sha256


def render(output, conditions):
    output, conditions = Path(output), Path(conditions)
    run = read_json(output / "run.json")
    records = read_json(output / "records.json")
    if run["conditions_sha256"] != sha256(conditions / "manifest.json"):
        raise ValueError("Report condition cache differs from inference")
    (output / "comparisons").mkdir(exist_ok=True)
    cards = []
    for index, frame in enumerate(run["source"]["frames"], 1):
        initial, edge = read_condition(conditions, frame)
        images = [("TIR", initial), ("clarity E (soft strength)",
                  Image.fromarray(np.rint(edge * 255).astype(np.uint8)).convert("RGB"))]
        for variant in run["variants"]:
            path = output / variant / "images" / f"{frame['stem']}_fake_B.png"
            if path.is_file():
                with Image.open(path) as im:
                    images.append((variant, im.convert("RGB")))
        w, h = initial.size
        sheet = Image.new("RGB", (w * len(images), h + 30), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (label, image) in enumerate(images):
            draw.text((column * w + 6, 7), label, fill="black")
            sheet.paste(image, (column * w, 30))
        relative = f"comparisons/{frame['stem']}.png"
        sheet.save(output / relative)
        cards.append(f'<article><h3>{index:02d} | {html.escape(frame["filename"])}</h3>'
                     f'<a href="{html.escape(relative, quote=True)}"><img loading="lazy" '
                     f'src="{html.escape(relative, quote=True)}"></a></article>')
    config = html.escape(json.dumps(run["config"], ensure_ascii=False, indent=2))
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>D0 / D1</title>'
    page += '<style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}article{background:white;padding:12px;margin:16px 0}img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style>'
    page += '<h1>D：预训练扩散基础对照与清晰轮廓控制</h1>'
    page += '<p>D0：红外 img2img，ControlNet 残差权重 0；D1：同一管线，增加未经二值化的 clarity_v1 E。每张、每个版本均重置 seed42。</p>'
    page += '<p>重点逐张检查人车位置、数量、外轮廓和细肢体，再看颜色、纹理和割裂。RGB 看起来更真实不等于忠实保留输入。</p>'
    page += '<p>本轮为零训练预览；借用 RGB 硬边缘预训练控制分支。没有独立语义特征注入、GMM 颜色或轮廓置信度。既有语义模型仅参与 E 的重要性融合。</p>'
    page += '<p>点击图片查看原尺寸对照。所有所选帧均列出；单帧随机种子不构成多随机种子统计。</p>'
    flags = sum(item.get("safety_flagged") is True for item in records.values())
    page += f'<p>已记录输出：{len(records)}，安全检查过滤：{flags}。若出现过滤，先检查记录，不能将黑图当作模型正常输出来评价。</p>'
    page += f'<details><summary>固定配置</summary><pre>{config}</pre></details>' + ''.join(cards) + '</html>'
    (output / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    args = parser.parse_args()
    render(args.output_dir, args.conditions)
