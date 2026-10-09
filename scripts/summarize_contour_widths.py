#!/usr/bin/env python3
"""Summarize paired ordinal width reviews, never pixel boundary accuracy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = ('1', '0.75', '0.5')
METHODS = ('C2b', 'C2c')
METRICS = ('missing', 'spread', 'extra')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--pack-dir', type=Path, default=ROOT/'runs/edge_audit/contour_width_review_v1')
    args = parser.parse_args()
    raw = args.ratings.read_bytes()
    exported = json.loads(raw)
    manifest = json.loads((args.pack_dir/'manifest.json').read_text())
    if exported.get('pack_id') != manifest['pack_id']:
        raise ValueError('Ratings package ID does not match the width preview')
    items = {i['id']: i for i in manifest['items']}
    ratings = exported['ratings']
    unknown = set(ratings) - set(items)
    if unknown:
        raise ValueError(f'Unknown target IDs: {sorted(unknown)}')
    fields = [f'{m}_{w}_{metric}' for m in METHODS for w in WIDTHS for metric in METRICS]
    for target, rating in ratings.items():
        for field in fields:
            if rating.get(field, '') not in ('', '0', '1', '2', 'u'):
                raise ValueError(f'Invalid grade: {target}/{field}')
    complete = lambda r: all(r.get(f) in ('0', '1', '2') for f in fields)
    eligible = [i for i, r in ratings.items() if r.get('visibility') in ('clear', 'weak')]
    primary = [i for i in eligible if complete(ratings[i])]
    uncertain = [i for i, r in ratings.items() if r.get('visibility') == 'uncertain' and complete(r)]
    cohorts = {'primary_visible_complete': primary, 'uncertain_complete_separate': uncertain,
               'primary_small_short_side_le_12': [i for i in primary if items[i]['short_side'] <= 12],
               'primary_large_short_side_ge_32': [i for i in primary if items[i]['short_side'] >= 32]}
    stats = {}
    for name, ids in cohorts.items():
        counts, paired = {}, {}
        for method in METHODS:
            for width in WIDTHS:
                key = f'{method}_{width}'
                counts[key] = {metric: dict(Counter(ratings[i][f'{key}_{metric}'] for i in ids)) for metric in METRICS}
                if width == '1':
                    continue
                paired[key] = {}
                for metric in METRICS:
                    changes = [int(ratings[i][f'{key}_{metric}']) - int(ratings[i][f'{method}_1_{metric}']) for i in ids]
                    paired[key][metric] = {'lower_grade': sum(v < 0 for v in changes),
                                           'same_grade': sum(v == 0 for v in changes),
                                           'higher_grade': sum(v > 0 for v in changes)}
        stats[name] = {'n': len(ids), 'ids': ids, 'grade_counts': counts, 'paired_vs_width_1': paired}
    flags = []
    for target in primary:
        r = ratings[target]
        reasons = []
        if any(r[f'{m}_0.5_missing'] == '2' for m in METHODS):
            reasons.append('narrow width still has severe missing boundary grade')
        if any(r[f'{m}_0.5_extra'] == '2' for m in METHODS):
            reasons.append('narrow width still has severe extra-edge grade')
        if r.get('notes'):
            reasons.append(r['notes'])
        if reasons:
            flags.append({'id': target, 'reasons': reasons})
    report = {'pack_id': manifest['pack_id'], 'ratings_source': str(args.ratings.resolve()),
              'ratings_sha256': hashlib.sha256(raw).hexdigest(), 'reviewed_count': len(ratings),
              'visibility_counts': dict(Counter(r.get('visibility', '') for r in ratings.values())),
              'visible_but_incomplete_ids': sorted(set(eligible)-set(primary)), 'cohorts': stats,
              'review_flags': flags,
              'limitation': 'Paired human ordinal grades from selected development targets, not accuracy, recall or independent model evaluation. Size groups are bbox proxies.'}
    (args.pack_dir/'width_review_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    labels = {'missing': '主要边界缺失', 'spread': '边界扩散', 'extra': '疑似额外边缘'}
    lines = ['# 软轮廓宽度评级汇总', '', f'读取文件：`{args.ratings.resolve()}`。包ID校验通过。', '',
             f'共{len(ratings)}项；主比较{len(primary)}项，原图可辨且18项评级完整。',
             f'无法确定但评级完整的{len(uncertain)}项单独保留；无对应目标及未完成项不计入主比较。', '',
             '评级是0无/轻微、1中等、2严重，数值越低越好；不合成总分，不计算像素准确率。', '',
             '## 主比较', '', '| 方法 | 宽度 | 缺失 0/1/2 | 扩散 0/1/2 | 额外边缘 0/1/2 |',
             '|---|---|---|---|---|']
    for method in METHODS:
        for width in WIDTHS:
            row = stats['primary_visible_complete']['grade_counts'][f'{method}_{width}']
            values = [' / '.join(str(row[metric].get(str(g), 0)) for g in range(3)) for metric in METRICS]
            lines.append(f'| {method} | {width} | '+ ' | '.join(values)+' |')
    lines.extend(['', '## 宽度0.5相对1.0：同目标配对比较', '', '| 方法 | 维度 | 等级降低 | 不变 | 等级升高 |', '|---|---|---|---|---|'])
    for method in METHODS:
        for metric in METRICS:
            p = stats['primary_visible_complete']['paired_vs_width_1'][f'{method}_0.5'][metric]
            lines.append(f"| {method} | {labels[metric]} | {p['lower_grade']} | {p['same_grade']} | {p['higher_grade']} |")
    lines.extend(['', '## 需要复核', ''])
    for flag in flags:
        lines.append(f"- {flag['id']}：{'；'.join(flag['reasons'])}")
    lines.extend(['', '## 判断范围', '',
                  '0.5可以作为后续候选；主比较中没有评级升高，不等于证明不存在损伤。样本少，无法确定目标被排除，观察者知道参数，存在选择与判断偏差。',
                  '变细、变清晰可降低人工缺失等级，不代表检测器新增了真实边界。C2b的硬候选不随宽度变化；C2c会重新计算纹理密度与重要性权重。',
                  '疑似额外边缘等级降低也不等于删除了错误候选；可能只是尾部扩散减少、视觉干扰减轻。',
                  '建议复核严重失败例和独立固定抽取的正常目标，再决定是否接受轮廓候选。现有训练保持原配置；确认候选后，用独立实验名进行相同80epoch训练和正常/置零条件对照。'])
    (args.pack_dir/'width_review_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'reviewed': len(ratings), 'primary': len(primary), 'uncertain_separate': len(uncertain),
                      'primary_counts': stats['primary_visible_complete']['grade_counts'],
                      'paired': stats['primary_visible_complete']['paired_vs_width_1'], 'flags': flags}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
