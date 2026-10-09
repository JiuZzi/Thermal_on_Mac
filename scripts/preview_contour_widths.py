#!/usr/bin/env python3
"""Fixed-sample width preview. Boxes are locations, never boundary ground truth."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (1.0, 0.75, 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-pack', type=Path, default=ROOT / 'runs/edge_audit/important_contour_review_v1')
    parser.add_argument('--ratings', type=Path, default=ROOT / 'scripts/contour_review_ratings.json')
    parser.add_argument('--processed-dir', type=Path, default=ROOT / 'FLIR_datasets/trainB')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/edge_audit/contour_width_review_v1')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output folder to preserve previous reviews')
    base = json.loads((args.base_pack / 'review_manifest.json').read_text())
    ratings = json.loads(args.ratings.read_text())
    if ratings['pack_id'] != base['pack_id']:
        raise ValueError('Ratings and fixed sample package do not match')
    reviewed_ids = set(ratings['ratings'])
    items = [dict(item, previously_reviewed=item['id'] in reviewed_ids)
             for item in base['items']
             if item.get('priority') or item['category'] == 'building_scene' or item['id'] in reviewed_ids]
    sys.path.insert(0, str(ROOT / 'pytorch-CycleGAN-and-pix2pix'))
    from models.conditioned_generator import ConditionedGenerator
    from models.saliency_edges import make_adaptive_saliency_edge, soft_multi_at_sigma, structure_texture_importance

    generators = {(m, w): ConditionedGenerator(nn.Identity(), mode, edge_soft_width=w, fusion_mode='direct')
                  for m, mode in [('C2b', 'soft_multi'), ('C2c', 'soft_saliency')] for w in WIDTHS}
    out = args.output_dir
    (out / 'frames').mkdir(parents=True)
    (out / 'targets').mkdir()
    cards = []
    concentration = []
    for number, sample in enumerate(base['selection']['samples'], 1):
        path = args.processed_dir / sample['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest() != base['source_hashes'][sample['filename']]:
            raise ValueError(f'Source image changed: {path}')
        original = Image.open(path).convert('RGB').convert('L')
        if base['geometry'] == 'center_crop_256':
            original = original.crop((52, 16, 308, 272))
        gray = np.asarray(original).copy()
        tir = torch.from_numpy(gray.astype(np.float32) / 127.5 - 1)[None, None]
        normalized = np.clip((tir[0, 0].numpy() + 1) / 2, 0, 1)
        maps = {}
        for (method, width), generator in generators.items():
            edge, reserved = generator.make_condition(tir)
            assert torch.count_nonzero(reserved).item() == 0
            maps[f'{method}_{width:g}'] = edge[0, 0].numpy()
        prefix = f'{number:02d}'
        with np.load(args.base_pack / 'frames' / f'{prefix}_maps.npz') as previous:
            maps['C1'] = previous['C1'].copy()
            maps['candidate'] = previous['candidate'].copy()
            for method in ('C2b', 'C2c'):
                np.testing.assert_allclose(maps[f'{method}_1'], previous[method], rtol=0, atol=1e-7)
        _, importance, fine0, coarse0 = make_adaptive_saliency_edge(normalized, (.12, .16, .20), .5, 1., (.7, 1., 1.6), .5)
        _, clutter0 = structure_texture_importance(normalized, fine0, coarse0)
        for width in WIDTHS:
            fine_maps = [soft_multi_at_sigma(normalized, s, (.12, .16, .20), .5, width) for s in (.7, 1., 1.6)]
            fine = (fine_maps[0] + fine_maps[1]) / 2
            coarse = (fine_maps[1] + fine_maps[2]) / 2
            fixed = (importance * fine + (1 - importance) * .5 * coarse) * (1 - .4 * clutter0)
            maps[f'C2c_fixed_{width:g}'] = np.clip(np.maximum(fixed, .25 * fine), 0, 1)
        np.testing.assert_allclose(maps['C2c_fixed_1'], maps['C2c_1'], rtol=0, atol=1e-7)
        for width in (.75, .5):
            assert np.all(maps[f'C2b_{width:g}'] <= maps['C2b_1'] + 1e-7)
        np.savez_compressed(out / 'frames' / f'{prefix}_maps.npz', **maps)

        def grid(crop, kind, gain=1, fixed=False):
            raw = original.crop(crop).convert('RGB')
            scale = min(6, max(1, math.ceil(180 / raw.height)))
            size = (raw.width * scale, raw.height * scale)
            methods = ('C2c_fixed',) if fixed else ('C2b', 'C2c')
            sheet = Image.new('RGB', (5 * size[0], len(methods) * (size[1] + 36)), 'white')
            draw = ImageDraw.Draw(sheet)
            for row, method in enumerate(methods):
                keys = ['TIR', 'C1', *[f'{method}_{w:g}' for w in WIDTHS]]
                for col, key in enumerate(keys):
                    if key == 'TIR':
                        image = original.convert('RGB')
                    elif kind == 'gray':
                        image = Image.fromarray(np.rint(maps[key] * 255).astype(np.uint8)).convert('RGB')
                    else:
                        alpha = np.clip(maps[key] * gain, 0, 1)[..., None] * .8
                        source = np.repeat(gray[..., None], 3, axis=2)
                        image = Image.fromarray(np.clip(source * (1 - alpha) + np.array([255, 55, 45]) * alpha, 0, 255).astype(np.uint8))
                    image = image.crop(crop).resize(size, Image.Resampling.NEAREST)
                    xy = (col * size[0], row * (size[1] + 36))
                    sheet.paste(image, xy)
                    draw.text((xy[0] + 3, xy[1] + size[1] + 3), key, fill='black')
            return sheet

        for item in [i for i in items if i['frame_number'] == number]:
            target = item['id']
            crop = tuple(item['crop'])
            for suffix, kind, gain, fixed in [('overlay', 'overlay', 1, False), ('gray', 'gray', 1, False),
                                               ('gain4', 'overlay', 4, False), ('fixed', 'overlay', 1, True)]:
                grid(crop, kind, gain, fixed).save(out / 'targets' / f'{target}_{suffix}.png')
            original.crop(crop).save(out / 'targets' / f'{target}_tir.png')
            box = item['box']
            bw, bh = box[2] - box[0], box[3] - box[1]
            item['short_side'] = min(bw, bh)
            # This only describes map concentration, not object interior or precision.
            region = (slice(crop[1], crop[3]), slice(crop[0], crop[2]))
            support = maps['candidate'][region] > 0
            for key, values in maps.items():
                if key.startswith(('C2b_', 'C2c_')):
                    patch = values[region]
                    total = float(patch.sum())
                    concentration.append({'id': target, 'method': key,
                                          'off_candidate_mass_fraction': float(patch[~support].sum()) / total if total else None,
                                          'meaning': 'response mass outside fixed hard-candidate pixels; not false-edge rate'})
            previous = ratings['ratings'].get(target, {})
            rows = []
            for method in ('C2b', 'C2c'):
                for width in WIDTHS:
                    fields = ''.join(f'<label>{label}<select data-field="{method}_{width:g}_{name}"><option value="">未审查</option><option value="0">0 无/轻微</option><option value="1">1 中等</option><option value="2">2 严重</option><option value="u">无法判断</option></select></label>'
                                     for name, label in [('missing', '主要边界缺失'), ('spread', '边界扩散'), ('extra', '疑似额外边缘')])
                    rows.append(f'<div class="rating"><b>{method} w={width:g}</b>{fields}</div>')
            cards.append(f'''<article data-id="{target}" data-reviewed="{str(item['previously_reviewed']).lower()}" data-scene="{str(item['category']=='building_scene').lower()}" data-short="{min(bw,bh):.3f}">
<h2>{target} · {html.escape(item['category'])} · 框 {bw:.1f}×{bh:.1f} 像素</h2>
<p>{'裁剪截断：只评可见部分。' if item['truncated_by_preprocessing'] else '框只用于定位。'}此前可辨性：{html.escape(previous.get('visibility','未审查'))}</p>
<img src="targets/{target}_tir.png" alt="原始红外" loading="lazy"><p>原图可辨性 <select data-field="visibility"><option value="">未审查</option><option value="clear">清楚</option><option value="weak">弱但可辨</option><option value="uncertain">无法确定</option><option value="absent">无对应目标</option></select></p>
<details {'open' if item['previously_reviewed'] else ''}><summary>展开宽度对照：上排 C2b，下排 C2c</summary>
<p>每排：原图、C1 硬边、软边宽度 1.0、0.75、0.5；裁剪与放大完全一致。</p>
<img src="targets/{target}_overlay.png" alt="统一强度叠加" loading="lazy"><h3>纯灰度边缘图：黑=0、白=1，无逐图归一化</h3>
<img src="targets/{target}_gray.png" alt="纯边缘强度" loading="lazy">
<details><summary>统一增亮 ×4（仅辅助找弱边）</summary><img src="targets/{target}_gain4.png" loading="lazy"></details>
<details><summary>辅助诊断：固定 w=1 的 C2c 权重，只改宽度（不作为训练版本）</summary><img src="targets/{target}_fixed.png" loading="lazy"></details>
{''.join(rows)}<p>备注：具体缺失位置、过宽区域、独立多余线条；真实内部结构不算错误。</p><textarea data-field="notes"></textarea></details></article>''')
        print(f'Generated fixed frame {number}/40', flush=True)

    manifest = {'base_pack_id': base['pack_id'], 'widths': WIDTHS, 'geometry': base['geometry'], 'items': items,
                'selection': base['selection'], 'source_hashes': base['source_hashes'],
                'parameters': dict(base['parameters'], soft_width=list(WIDTHS)),
                'baseline_reproduction': 'C2b/C2c w=1 matched original maps within absolute 1e-7 on all frames',
                'C2c_note': 'coefficients fixed; importance/clutter recomputed with width. fixed row freezes w=1 weights.',
                'limitation': 'Development preview. No human boundary labels; no accuracy or model-quality inference.',
                'code_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), ROOT/'pytorch-CycleGAN-and-pix2pix/models/conditioned_generator.py', ROOT/'pytorch-CycleGAN-and-pix2pix/models/saliency_edges.py']}}
    pack_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    manifest['pack_id'] = pack_id
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out/'concentration_diagnostics.json').write_text(json.dumps(concentration, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>软轮廓宽度固定样本对照</title>
<style>body{font:16px/1.6 system-ui;background:#f3f5f8;color:#17212b;margin:0}header{position:sticky;top:0;background:white;padding:12px;z-index:1;border-bottom:1px solid #bbb}main{max-width:1400px;margin:auto;padding:18px}article{background:white;padding:20px;margin:20px 0;border:1px solid #ccd}img{max-width:100%;height:auto}select,button{padding:8px;margin:4px}summary{cursor:pointer;font-weight:bold}.rating{display:flex;flex-wrap:wrap;gap:8px;border-top:1px solid #ddd;padding:8px}textarea{width:95%;min-height:70px}label{display:inline-block}</style>
<header><b>软轮廓宽度对照</b> <span id="count"></span><select id="filter"><option value="reviewed">已审查的 11 个目标</option><option value="small">小目标：框短边≤12像素</option><option value="large">大目标：框短边≥32像素</option><option value="scene">建筑全图</option><option value="all">全部固定重点目标</option></select><button id="export">导出三项评级 JSON</button></header>
<main><h1>先查主要边界缺失，再查扩散与额外边缘</h1><p>范围来自原来固定的40张训练集图像。尺寸分组只是浏览辅助，不是语义标注。原图不可辨的目标不能据此判定漏检；内部红色不直接等于错误边缘。请先对大、小目标分别检查。</p><p>正式 C2c 的密度权重会随宽度变化；辅助行用于区分权重变化与单纯缩窄。此页没有改变训练配置。</p>__CARDS__</main>
<script>const packId=__ID__;const key='width-review-'+packId;let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
for(const card of document.querySelectorAll('article')){for(const field of card.querySelectorAll('[data-field]')){field.value=ratings[card.dataset.id]?.[field.dataset.field]||'';field.addEventListener('change',()=>{ratings[card.dataset.id]??={};ratings[card.dataset.id][field.dataset.field]=field.value;try{localStorage.setItem(key,JSON.stringify(ratings))}catch(e){alert('缓存保存失败，请及时导出JSON')}})}}
function filter(){const f=document.getElementById('filter').value;let count=0;for(const card of document.querySelectorAll('article')){const scene=card.dataset.scene==='true';const show=f==='reviewed'?card.dataset.reviewed==='true':f==='small'?!scene&&+card.dataset.short<=12:f==='large'?!scene&&+card.dataset.short>=32:f==='scene'?scene:!scene;card.hidden=!show;count+=show?1:0}document.getElementById('count').textContent='当前显示 '+count+' 项'}document.getElementById('filter').addEventListener('change',filter);filter();
document.getElementById('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({pack_id:packId,ratings},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='contour_width_ratings.json';a.click();URL.revokeObjectURL(url)});
</script></html>'''.replace('__CARDS__', ''.join(cards)).replace('__ID__', json.dumps(pack_id))
    (out/'index.html').write_text(page, encoding='utf-8')
    print(f'Output: {out / "index.html"}; {len(items)} fixed items')


if __name__ == '__main__':
    main()
