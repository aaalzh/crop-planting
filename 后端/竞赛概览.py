from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from 后端.发布治理 import get_release_status
from 后端.环境桥接 import load_env_scenario_library
from 后端.反馈回流 import get_feedback_training_status

MODULE_LABELS = {
    ('price', 'mae'): '价格 MAE',
    ('price', 'rmse'): '价格 RMSE',
    ('price', 'mape'): '价格 MAPE',
    ('yield', 'mae'): '产量 MAE',
    ('yield', 'rmse'): '产量 RMSE',
    ('yield', 'mape'): '产量 MAPE',
    ('cost', 'mae'): '成本 MAE',
    ('cost', 'rmse'): '成本 RMSE',
    ('cost', 'mape'): '成本 MAPE',
}
BUSINESS_LABELS = {
    'profit_mae': '利润 MAE',
    'topk_avg_profit': 'Top-K 平均利润',
    'ndcg_at_k': 'NDCG@K',
    'hit_rate_at_k': 'HitRate@K',
}
SPOTLIGHT_ORDER = [
    'profit_mae',
    'yield.rmse',
    'yield.mape',
    'price.rmse',
    'hit_rate_at_k',
    'ndcg_at_k',
]


def _to_abs(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number:
        return None
    return number


def _overview_path(root: Path, config: dict) -> Path:
    feedback_cfg = config.get('feedback', {}) if isinstance(config, dict) else {}
    target = str(feedback_cfg.get('competition_overview_file', '输出/闭环/竞赛概览.json'))
    return _to_abs(root, target)


def _release_file(item: dict, name: str) -> Optional[Path]:
    release_dir = str(item.get('release_dir', '')).strip()
    if not release_dir:
        return None
    path = Path(release_dir) / name
    return path if path.exists() else None


def _load_release_bundle(item: dict) -> dict:
    release_report_path = _release_file(item, '发布报告.json') or _release_file(item, 'release_report.json')
    backtest_report_path = _release_file(item, '回测报告.json')
    training_report_path = _release_file(item, '模型训练报告.json')
    return {
        'release_report_path': release_report_path.as_posix() if release_report_path else None,
        'backtest_report_path': backtest_report_path.as_posix() if backtest_report_path else None,
        'training_report_path': training_report_path.as_posix() if training_report_path else None,
        'release_report': _read_json(release_report_path) if release_report_path else {},
        'backtest_report': _read_json(backtest_report_path) if backtest_report_path else {},
        'training_report': _read_json(training_report_path) if training_report_path else {},
    }


def _reference_release(release: dict) -> dict:
    champion = release.get('champion') or {}
    challenger = release.get('challenger') or {}
    archived = release.get('archived') or []

    if challenger.get('run_id') and champion.get('run_id'):
        return {
            'mode': 'challenger_vs_champion',
            'primary': challenger,
            'reference': champion,
            'primary_label': '当前挑战者',
            'reference_label': '当前冠军',
            'report_owner': challenger,
        }

    if champion.get('run_id'):
        preferred = None
        fallback = None
        for item in archived:
            if not isinstance(item, dict):
                continue
            if str(item.get('run_id') or '').strip() == str(champion.get('run_id') or '').strip():
                continue
            if fallback is None:
                fallback = item
            if item.get('allowed') is True or str(item.get('summary') or '').strip().lower() == 'pass':
                preferred = item
                break
        reference = preferred or fallback
        if reference:
            return {
                'mode': 'champion_vs_previous',
                'primary': champion,
                'reference': reference,
                'primary_label': '当前冠军',
                'reference_label': '上一任冠军' if reference.get('allowed') is True else '历史版本',
                'report_owner': champion,
            }

    return {}


def _comparison_metric(*, key: str, label: str, direction: str, primary_value: Any, reference_value: Any, passed: Any = None) -> dict:
    primary = _safe_float(primary_value)
    reference = _safe_float(reference_value)
    delta = None if primary is None or reference is None else float(primary - reference)
    if primary is None or reference is None or abs(reference) <= 1e-12:
        improvement_ratio = None
        trend = 'flat'
    elif str(direction) == 'lower_is_better':
        improvement_ratio = float((reference - primary) / abs(reference))
        trend = 'better' if primary < reference - 1e-12 else 'worse' if primary > reference + 1e-12 else 'flat'
    else:
        improvement_ratio = float((primary - reference) / abs(reference))
        trend = 'better' if primary > reference + 1e-12 else 'worse' if primary < reference - 1e-12 else 'flat'
    return {
        'key': key,
        'label': label,
        'direction': direction,
        'primary_value': primary,
        'reference_value': reference,
        'delta': delta,
        'improvement_ratio': improvement_ratio,
        'trend': trend,
        'passed': bool(passed) if passed is not None else None,
    }


def _collect_compare_metrics(report: dict) -> tuple[list[dict], list[dict]]:
    gate = report.get('gate', {}) if isinstance(report, dict) else {}
    module_metrics = []
    for item in gate.get('module_checks', []) or []:
        scope = str(item.get('scope') or '').strip().lower()
        metric = str(item.get('metric') or '').strip().lower()
        label = MODULE_LABELS.get((scope, metric), f'{scope}.{metric}')
        module_metrics.append(
            _comparison_metric(
                key=f'{scope}.{metric}',
                label=label,
                direction=str(item.get('direction') or 'lower_is_better'),
                primary_value=item.get('challenger'),
                reference_value=item.get('champion'),
                passed=item.get('passed'),
            )
        )

    business_metrics = []
    for item in gate.get('business_checks', []) or []:
        metric = str(item.get('metric') or '').strip()
        label = BUSINESS_LABELS.get(metric, metric)
        business_metrics.append(
            _comparison_metric(
                key=metric,
                label=label,
                direction=str(item.get('direction') or 'higher_is_better'),
                primary_value=item.get('challenger'),
                reference_value=item.get('champion'),
                passed=item.get('passed'),
            )
        )
    return module_metrics, business_metrics


def _yearly_series(backtest_report: dict) -> dict:
    business_year = (((backtest_report.get('business_metrics') or {}).get('test') or {}).get('by_year') or {})
    yield_year = (((((backtest_report.get('robustness') or {}).get('test') or {}).get('yield') or {}).get('by_year')) or {})
    years = sorted({str(y) for y in list(business_year.keys()) + list(yield_year.keys())}, key=lambda x: int(x))
    return {
        'years': years,
        'topk_avg_profit': [_safe_float((business_year.get(y) or {}).get('topk_avg_profit')) for y in years],
        'ndcg_at_k': [_safe_float((business_year.get(y) or {}).get('ndcg_at_k')) for y in years],
        'hit_rate_at_k': [_safe_float((business_year.get(y) or {}).get('hit_rate_at_k')) for y in years],
        'yield_mae': [_safe_float((((yield_year.get(y) or {}).get('all') or {}).get('mae'))) for y in years],
        'yield_rmse': [_safe_float((((yield_year.get(y) or {}).get('all') or {}).get('rmse'))) for y in years],
    }


def _pick_spotlight_metrics(module_metrics: list[dict], business_metrics: list[dict]) -> list[dict]:
    merged = {item.get('key'): item for item in [*business_metrics, *module_metrics] if item.get('key')}
    picked = [merged[key] for key in SPOTLIGHT_ORDER if key in merged]
    for item in [*business_metrics, *module_metrics]:
        if item not in picked and len(picked) < 6:
            picked.append(item)
    return picked[:6]


def _build_highlights(compare_meta: dict, module_metrics: list[dict], business_metrics: list[dict], yearly_primary: dict, report: dict) -> list[str]:
    highlights: list[str] = []
    by_key = {item.get('key'): item for item in [*module_metrics, *business_metrics]}
    profit = by_key.get('profit_mae')
    if profit and profit.get('primary_value') is not None and profit.get('reference_value') is not None:
        highlights.append(
            f"{BUSINESS_LABELS['profit_mae']} 从 {profit['reference_value']:.0f} 降到 {profit['primary_value']:.0f}。"
        )
    y_rmse = by_key.get('yield.rmse')
    if y_rmse and y_rmse.get('primary_value') is not None and y_rmse.get('reference_value') is not None:
        highlights.append(
            f"产量 RMSE 从 {y_rmse['reference_value']:.3f} 调整到 {y_rmse['primary_value']:.3f}。"
        )
    profits = [value for value in yearly_primary.get('topk_avg_profit', []) if value is not None]
    years = yearly_primary.get('years', [])
    if profits and years:
        highlights.append(
            f"{years[0]}-{years[-1]} 年测试集 Top-K 平均利润稳定在 {min(profits):.0f} 到 {max(profits):.0f} 区间。"
        )
    shadow = report.get('shadow', {}) if isinstance(report, dict) else {}
    if shadow.get('ok') is True:
        highlights.append(
            f"shadow replay 回放 {int(shadow.get('attempted_events', 0) or 0)} 条事件，Top1 一致率 {float(shadow.get('top1_match_rate', 0.0) or 0.0):.2f}。"
        )
    improved = sum(1 for item in [*module_metrics, *business_metrics] if item.get('trend') == 'better')
    regressed = sum(1 for item in [*module_metrics, *business_metrics] if item.get('trend') == 'worse')
    highlights.append(f"{compare_meta.get('primary_label', '当前版本')} 相对 {compare_meta.get('reference_label', '参考版本')} 改善 {improved} 项，回退 {regressed} 项。")
    return highlights[:5]


def _build_release_comparison(root: Path, release: dict) -> dict:
    compare_meta = _reference_release(release)
    if not compare_meta:
        return {
            'available': False,
            'reason': 'no_reference_release',
            'module_metrics': [],
            'business_metrics': [],
            'spotlight_metrics': [],
            'highlights': [],
            'yearly_backtest': {},
        }

    primary_item = compare_meta['primary']
    reference_item = compare_meta['reference']
    primary_bundle = _load_release_bundle(primary_item)
    reference_bundle = _load_release_bundle(reference_item)
    report = primary_bundle.get('release_report') or {}
    module_metrics, business_metrics = _collect_compare_metrics(report)
    spotlight_metrics = _pick_spotlight_metrics(module_metrics, business_metrics)
    primary_yearly = _yearly_series(primary_bundle.get('backtest_report') or {})
    reference_yearly = _yearly_series(reference_bundle.get('backtest_report') or {})

    highlights = _build_highlights(compare_meta, module_metrics, business_metrics, primary_yearly, report)
    improved = sum(1 for item in [*module_metrics, *business_metrics] if item.get('trend') == 'better')
    regressed = sum(1 for item in [*module_metrics, *business_metrics] if item.get('trend') == 'worse')

    return {
        'available': True,
        'mode': compare_meta.get('mode'),
        'primary_label': compare_meta.get('primary_label'),
        'reference_label': compare_meta.get('reference_label'),
        'primary': {
            'run_id': primary_item.get('run_id'),
            'status': primary_item.get('status'),
            'summary': primary_item.get('summary'),
            'manifest_path': primary_item.get('manifest_path'),
            'release_report_path': primary_bundle.get('release_report_path'),
            'backtest_report_path': primary_bundle.get('backtest_report_path'),
        },
        'reference': {
            'run_id': reference_item.get('run_id'),
            'status': reference_item.get('status'),
            'summary': reference_item.get('summary'),
            'manifest_path': reference_item.get('manifest_path'),
            'release_report_path': reference_bundle.get('release_report_path'),
            'backtest_report_path': reference_bundle.get('backtest_report_path'),
        },
        'summary': {
            'improved_count': improved,
            'regressed_count': regressed,
            'smoke_ok': (report.get('smoke') or {}).get('ok'),
            'shadow_ok': (report.get('shadow') or {}).get('ok'),
            'shadow_attempted_events': (report.get('shadow') or {}).get('attempted_events'),
        },
        'module_metrics': module_metrics,
        'business_metrics': business_metrics,
        'spotlight_metrics': spotlight_metrics,
        'highlights': highlights,
        'yearly_backtest': {
            'years': primary_yearly.get('years', []),
            'primary': {
                'label': compare_meta.get('primary_label'),
                'run_id': primary_item.get('run_id'),
                **primary_yearly,
            },
            'reference': {
                'label': compare_meta.get('reference_label'),
                'run_id': reference_item.get('run_id'),
                **reference_yearly,
            },
        },
    }


def build_competition_overview(root: Path, config: dict, *, save: bool = True) -> dict:
    root = root.resolve()
    release = get_release_status(root=root, config=config)
    feedback_training = get_feedback_training_status(root=root, config=config, refresh=True)
    env_scenarios = load_env_scenario_library(root=root, config=config, rebuild_if_missing=True)
    champion = release.get('champion') or {}
    comparison = _build_release_comparison(root=root, release=release)

    checklist = {
        'release_bundle_ready': bool(champion.get('manifest_path')),
        'champion_online': str(champion.get('status') or '').strip().lower() == 'champion',
        'shadow_replay_ready': champion.get('shadow_ok') is True,
        'feedback_logging_ready': int(feedback_training.get('inference_event_count', 0) or 0) > 0,
        'feedback_dataset_ready': int(feedback_training.get('labeled_sample_count', 0) or 0) > 0,
        'env_scenario_library_ready': int(env_scenarios.get('scenario_count', 0) or 0) > 0,
    }

    key_artifacts = {
        'competition_overview_file': _overview_path(root, config).as_posix(),
        'champion_manifest': champion.get('manifest_path'),
        'champion_release_dir': champion.get('release_dir'),
        'champion_backtest_report': comparison.get('primary', {}).get('backtest_report_path') if comparison.get('available') else None,
        'feedback_training_summary': feedback_training.get('training_summary_file'),
        'feedback_training_samples': feedback_training.get('training_sample_file'),
        'env_scenario_file': env_scenarios.get('scenario_file'),
    }

    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'overview_file': _overview_path(root, config).as_posix(),
        'checklist': checklist,
        'release': release,
        'comparison': comparison,
        'feedback_training': feedback_training,
        'env_scenarios': {
            'scenario_file': env_scenarios.get('scenario_file'),
            'scenario_count': env_scenarios.get('scenario_count'),
            'scenarios_per_crop': env_scenarios.get('scenarios_per_crop'),
            'topk': env_scenarios.get('topk'),
        },
        'key_artifacts': key_artifacts,
        'demo_commands': {
            'status': 'python 后端/训练/闭环演示.py --action status --refresh-feedback',
            'train_release': 'python 后端/训练/闭环高精度训练.py --run-id <run_id>',
            'rollback': 'python 后端/训练/闭环演示.py --action rollback --run-id <run_id>',
        },
    }

    if save:
        _write_json(_overview_path(root, config), payload)
    return payload


def get_competition_overview(root: Path, config: dict, *, refresh: bool = False) -> dict:
    path = _overview_path(root.resolve(), config)
    if refresh or not path.exists():
        return build_competition_overview(root=root, config=config, save=True)
    return json.loads(path.read_text(encoding='utf-8'))
