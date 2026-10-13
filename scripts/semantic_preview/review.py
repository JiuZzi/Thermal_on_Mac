"""Fixed-strength images, review crops and offline human ratings."""

import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inference import CLASS_NAMES, CLASS_LABELS, PALETTE

FOCUS = {"person": (8,), "vehicles": (9, 10, 11, 12, 13), "building": (3,), "vegetation": (6,)}
CAPTIONS = {"person": "行人软响应", "vehicles": "车辆组软响应", "building": "建筑软响应", "vegetation": "植被软响应"}


def score_maps(result):
    return {name: result["probabilities"][list(ids)].sum(0) for name, ids in FOCUS.items()}


def uint8_image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def label_image(labels):
    if labels.ndim != 2 or np.any(labels >= len(PALETTE)):
        raise ValueError("Invalid FMB categorical labels")
    return Image.fromarray(PALETTE[labels])


def red_overlay(gray, score, alpha=0.65):
    base = np.repeat(gray[..., None], 3, -1).astype(np.float32) / 255
    return uint8_image(base * (1 - alpha * score[..., None]) + np.array([1, 0, 0]) * alpha * score[..., None])


def save_frame(gray, predictions, directory, prefix):
    directory = Path(directory)
    Image.fromarray(gray).save(directory / f"{prefix}_TIR.png")
    outputs = {}
    for mode, result in predictions.items():
        colors = np.asarray(label_image(result["labels"]), dtype=np.float32) / 255
        base = np.repeat(gray[..., None], 3, -1).astype(np.float32) / 255
        files = {"labels": f"{prefix}_{mode}_labels.png", "semantic": f"{prefix}_{mode}_semantic.png",
                 "overlay": f"{prefix}_{mode}_overlay.png", "entropy": f"{prefix}_{mode}_entropy.png",
                 "max_score": f"{prefix}_{mode}_max_score.png"}
        Image.fromarray(result["labels"]).save(directory / files["labels"])
        label_image(result["labels"]).save(directory / files["semantic"])
        uint8_image(base * 0.55 + colors * 0.45).save(directory / files["overlay"])
        uint8_image(result["normalized_entropy"]).save(directory / files["entropy"])
        uint8_image(result["max_score"]).save(directory / files["max_score"])
        for name, values in score_maps(result).items():
            files[name] = f"{prefix}_{mode}_{name}.png"
            files[name + "_overlay"] = f"{prefix}_{mode}_{name}_overlay.png"
            uint8_image(values).save(directory / files[name])
            red_overlay(gray, values).save(directory / files[name + "_overlay"])
        outputs[mode] = files
    return outputs


def render_crop(gray, predictions, crop, destination, zoom=4):
    x0, y0, x1, y1 = crop
    crop_gray = gray[y0:y1, x0:x1]
    h, w = crop_gray.shape
    width = max(150, w * zoom)
    height = h * zoom + 28
    sheet = Image.new("RGB", (width * 7, height * len(predictions)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (mode, result) in enumerate(predictions.items()):
        labels = result["labels"][y0:y1, x0:x1]
        colors = np.asarray(label_image(labels), dtype=np.float32) / 255
        base = np.repeat(crop_gray[..., None], 3, -1).astype(np.float32) / 255
        panels = [("TIR", Image.fromarray(crop_gray).convert("RGB")),
                  ("semantic classes", label_image(labels)),
                  ("class overlay", uint8_image(base * .55 + colors * .45))]
        panels += [(name + " [0,1]", red_overlay(crop_gray, values[y0:y1, x0:x1]))
                   for name, values in score_maps(result).items()]
        for col, (name, image) in enumerate(panels):
            draw.text((col * width + 4, row * height + 4), f"{mode}: {name}", fill="black")
            enlarged = image.resize((w * zoom, h * zoom), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (col * width + (width - w * zoom) // 2, row * height + 25))
    sheet.save(destination)


def response_stats(probabilities, item):
    x0, y0, x1, y1 = item["box"]
    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    x1, y1 = int(np.ceil(x1)), int(np.ceil(y1))
    values = probabilities[:, y0:y1, x0:x1]
    mapping = {"person": (8,), "car": (9,), "truck": (10,), "bus": (11,), "motor": (12,), "bike": (13,)}
    ids = mapping.get(item["category"])
    stats = {"requested_classes": None, "mean_score_proxy": None, "max_score_proxy": None,
             "argmax_fraction_proxy": None}
    if ids and values.size:
        response = values[list(ids)].sum(0)
        stats.update(requested_classes="+".join(CLASS_NAMES[i] for i in ids),
                     mean_score_proxy=float(response.mean()), max_score_proxy=float(response.max()),
                     argmax_fraction_proxy=float(np.isin(values.argmax(0), ids).mean()))
    return stats


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def write_page(path, frames, targets, modes, pack_id, settings):
    esc = html.escape
    legend = "".join(f'<span><i style="background:rgb{tuple(map(int,color))}"></i>{i}: {esc(label)}</span>'
                     for i, (label, color) in enumerate(zip(CLASS_LABELS, PALETTE)))
    cards = []
    ratings = (("pending", "未审查"), ("recognized", "识别且基本覆盖"), ("partial", "识别但明显遗漏"),
               ("missed", "未识别"), ("wrong_class", "误识别为其他类别"), ("uncertain", "无法判断"))
    mode_labels = {"base": f'作者流程：短边至少 {settings["short_side"]}',
                   "detail": f'尺寸诊断：短边至少 {settings["detail_short_side"]}'}
    for frame in frames:
        prefix = frame["prefix"]
        grids = [f'<div class="grid"><figure><figcaption>原始红外</figcaption><img loading="lazy" src="frames/{prefix}_TIR.png"></figure>']
        for mode in modes:
            grids.append(f'<figure><figcaption>{mode_labels[mode]}：类别叠加</figcaption><img loading="lazy" src="frames/{frame["files"][mode]["overlay"]}"></figure>')
        grids.append('</div><details><summary>查看完整类别图、各类软响应和诊断图</summary>')
        for mode in modes:
            files = frame["files"][mode]
            grids.append(f'<h4>{mode_labels[mode]}</h4><div class="grid">')
            for key, title in (("semantic", "类别图（显示色不代表 RGB 着色）"), *CAPTIONS.items(),
                               ("entropy", "归一化类别熵：亮表示分布更分散"), ("max_score", "最大类别响应")):
                grids.append(f'<figure><figcaption>{esc(title)}</figcaption><img loading="lazy" src="frames/{files[key]}"></figure>')
            grids.append('</div>')
        grids.append('</details>')
        crops = []
        for target in targets:
            if target["filename"] != frame["filename"]:
                continue
            annotations = []
            for mode in modes:
                options = "".join(f'<option value="{value}">{label}</option>' for value, label in ratings)
                annotations.append(f'<label>{mode_labels[mode]} <select data-target="{esc(target["id"])}" data-mode="{mode}">{options}</select></label>')
            truncated = "；预处理截断，只评可见部分" if target.get("truncated_by_preprocessing") else ""
            stats = []
            for mode in modes:
                s = target[mode]
                if s["mean_score_proxy"] is not None:
                    stats.append(f'{mode}: {esc(s["requested_classes"])} 框内均值 {s["mean_score_proxy"]:.3f}、最大值 {s["max_score_proxy"]:.3f}；这些数值不是准确率')
            crops.append(f'<section class="target" id="{esc(target["id"])}"><h4>{esc(target["id"])} / {esc(target["category"])}{truncated}</h4>'
                         f'<a href="targets/{esc(target["panel"])}" target="_blank"><img loading="lazy" src="targets/{esc(target["panel"])}"></a>'
                         f'<p class="small">全图推理后裁剪，最近邻放大；{esc("；".join(stats))}</p>'
                         f'<div class="ratings">{"".join(annotations)}</div><textarea data-note="{esc(target["id"])}" placeholder="记录漏掉的部位、误分类或无法判断的原因"></textarea></section>')
        cards.append(f'<article id="frame-{prefix}"><h3>{prefix} / {esc(frame["group"])} / {esc(frame["filename"])}</h3>'
                     + "".join(grids) + f'<details open><summary>固定重点目标（{len(crops)} 个）</summary>{"".join(crops)}</details></article>')
    schema = json.dumps(dict(pack_id=pack_id, modes=modes,
                             targets=[{key: target[key] for key in ("id", "filename", "category", "group")} for target in targets]),
                        ensure_ascii=True).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLIR：冻结语义模型逐目标审查</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#18212c;margin:20px}main{max-width:1320px;margin:auto}
article,.notice,.toolbar{background:white;padding:18px;border:1px solid #ccd4de;border-radius:8px;margin:18px 0}
.notice{background:#fff4d7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
figure{margin:0}figure img{width:100%;image-rendering:auto}figcaption{font-weight:600}.target{border-top:1px solid #d8dee5;padding-top:12px;margin-top:20px}
.target img{max-width:100%;height:auto;image-rendering:pixelated}h3{overflow-wrap:anywhere}summary{cursor:pointer;color:#2459a0}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px}.legend i{display:inline-block;width:16px;height:16px;margin-right:4px;border:1px solid #888}
.ratings{display:flex;flex-wrap:wrap;gap:14px}select,button,input,textarea{font:inherit}textarea{display:block;width:95%;min-height:50px;margin:10px 0}
button{padding:8px 14px;cursor:pointer}.small{font-size:13px;color:#52606d}.toolbar{position:sticky;top:0;z-index:1}#status{white-space:pre-line}
</style><main><h1>冻结 SegFormer-B3：语义是否可用？</h1>
<div class="notice"><p>仅使用红外推理；trainB 开发审查。既有框只定位固定裁剪，不送入语义网络或 GAN。
语义图填满人体区域是正常的；这一页检查类别与区域，不评轮廓宽度。彩色类别图不是着色结果。</p>
<p>512 为作者预处理基线；更大尺寸（若启用）是同一权重的尺寸诊断，不是新训练模型。比较结果分开保留，不逐图挑选或融合。
软响应固定 [0,1] 显示，红色只表示强度。高类别分数不保证正确；类别熵及最大分数不是后续边界置信度。</p>
<p>先确认原图可辨认目标，再判断类别及可见区域覆盖。不可辨认/遮挡处选“无法判断”，不要把未知强行当错误。
没有像素语义真值，不计算或宣称 FLIR mIoU。植被类别不能直接判断每片叶子是否应删除。</p></div>
<div class="legend">__LEGEND__</div><p class="small">车辆组响应 = car + truck + bus + motorcycle + bicycle；框内统计仅为响应代理。
点击目标面板可打开原尺寸图。修改评分后导出 JSON；所有操作在本地页面完成。</p>
<div class="toolbar"><button id="export">导出审查 JSON</button> <label>导入此前评分 <input id="import" type="file" accept=".json"></label><div id="status"></div></div>
__CARDS__<script>
const schema=__SCHEMA__;const key='semantic_review_'+schema.pack_id;let results={};
try{results=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const valid=new Set(['pending','recognized','partial','missed','wrong_class','uncertain']);
const ids=new Set(schema.targets.map(x=>x.id));
function entry(id){if(!results[id])results[id]={ratings:{},note:''};return results[id]}
function update(){let lines=[];for(const mode of schema.modes){let c={};for(const v of valid)c[v]=0;for(const t of schema.targets)c[entry(t.id).ratings[mode]||'pending']++;
lines.push(mode+': 已审查 '+(schema.targets.length-c.pending)+'/'+schema.targets.length+'；基本覆盖 '+c.recognized+'；部分遗漏 '+c.partial+'；未识别 '+c.missed+'；误分类 '+c.wrong_class+'；无法判断 '+c.uncertain)}
document.getElementById('status').textContent=lines.join('\n');try{localStorage.setItem(key,JSON.stringify(results))}catch(e){}}
function restore(){document.querySelectorAll('select[data-target]').forEach(el=>{const v=entry(el.dataset.target).ratings[el.dataset.mode];el.value=valid.has(v)?v:'pending'});
document.querySelectorAll('textarea[data-note]').forEach(el=>el.value=entry(el.dataset.note).note||'');update()}
document.querySelectorAll('select[data-target]').forEach(el=>el.addEventListener('change',()=>{entry(el.dataset.target).ratings[el.dataset.mode]=el.value;update()}));
document.querySelectorAll('textarea[data-note]').forEach(el=>el.addEventListener('input',()=>{entry(el.dataset.note).note=el.value;update()}));
document.getElementById('export').onclick=()=>{const out={format:'semantic_review_v1',pack_id:schema.pack_id,modes:schema.modes,targets:schema.targets,results:results,
meaning:'Human development ratings; not pixel accuracy or contour correctness',exported_at:new Date().toISOString()};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='semantic_review_ratings.json';a.click();URL.revokeObjectURL(a.href)};
document.getElementById('import').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());if(d.format!=='semantic_review_v1'||d.pack_id!==schema.pack_id)throw Error('评分对应不同权重、样本或预处理');
if(!d.results||typeof d.results!=='object')throw Error('缺少评分');for(const [id,item]of Object.entries(d.results)){if(!ids.has(id)||!item.ratings)throw Error('目标列表不匹配');
for(const [mode,value]of Object.entries(item.ratings))if(!schema.modes.includes(mode)||!valid.has(value))throw Error('评分字段不匹配')}
results=d.results;restore()}catch(err){alert('无法导入：'+err.message)}};restore();
</script></main></html>'''
    Path(path).write_text(page.replace("__LEGEND__", legend).replace("__CARDS__", "\n".join(cards)).replace("__SCHEMA__", schema), encoding="utf-8")
