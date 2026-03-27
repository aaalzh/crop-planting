export const MODEL_TERM_LABELS = Object.freeze({
  MAE: "MAE（平均绝对误差）",
  RMSE: "RMSE（均方根误差）",
  MAPE: "MAPE（平均绝对百分比误差）",
  CAGR: "CAGR（复合年增长率）"
});

export const AXIS_LABELS = Object.freeze({
  date: "日期（YYYY-MM-DD）",
  year: "年份（年）"
});

export const UNIT_SYSTEM_NOTE =
  "单位口径说明：价格使用 ₹/公担（quintal），成本使用 ₹/公顷（hectare），产量使用 吨/公顷。价格与成本来自不同原始口径，收益值用于候选相对排序。";

const METRIC_META = Object.freeze({
  env_prob: {
    label: "环境适配概率",
    unit: "%",
    direction: "越大越好",
    source: "来源：环境分类模型概率输出"
  },
  prob_best: {
    label: "校准后最优概率",
    unit: "%",
    direction: "越大越好",
    source: "来源：概率校准器输出"
  },
  price_pred: {
    label: "预测价格",
    unit: "₹/公担（quintal）",
    direction: "越大通常越好",
    source: "来源：价格源数据单位文件，模型口径为 ₹/quintal"
  },
  cost_pred: {
    label: "预测成本",
    unit: "₹/公顷（hectare）",
    direction: "越小越好",
    source: "来源：成本数据单位文件，模型口径为 ₹/hectare"
  },
  yield: {
    label: "预测产量",
    unit: "吨/公顷（t/ha）",
    direction: "越大通常越好",
    source: "来源：产量数据单位文件，模型口径为 Tonnes/Hectare"
  },
  profit: {
    label: "预测收益",
    unit: "模型收益值",
    direction: "越大越好",
    source: "来源：price_pred * yield - cost_pred 的模型组合值"
  },
  risk: {
    label: "风险分",
    unit: "风险指数",
    direction: "越小越好",
    source: "来源：风险评估模块输出"
  },
  recommend_strength: {
    label: "推荐强度",
    unit: "/100",
    direction: "越大越值得优先考虑",
    source: "来源：环境适配、校准概率、风险安全度、利润缓冲与排序信号的展示型综合强度"
  },
  score: {
    label: "排序分（内部）",
    unit: "评分",
    direction: "越大越好",
    source: "来源：推荐器发布版综合打分公式，仅用于候选排序"
  },
  runtime_ms: {
    label: "请求耗时",
    unit: "毫秒（ms）",
    direction: "越小越好",
    source: "来源：服务端运行时统计"
  },
  margin_pct: {
    label: "毛利率",
    unit: "%",
    direction: "越大越好",
    source: "来源：收益与成本推导"
  },
  volatility_90d_pct: {
    label: "近90天波动率",
    unit: "%",
    direction: "越小越稳",
    source: "来源：价格历史序列统计"
  },
  yoy_change_pct: {
    label: "同比变化",
    unit: "%",
    direction: "需结合场景判断",
    source: "来源：价格序列同比计算"
  },
  cagr_pct: {
    label: MODEL_TERM_LABELS.CAGR,
    unit: "%/年",
    direction: "越大表示长期增速越快",
    source: "来源：历史序列长期趋势拟合"
  }
});

const RATIO_PERCENT_KEYS = new Set(["env_prob", "prob_best"]);
const VALUE_PERCENT_KEYS = new Set(["margin_pct", "volatility_90d_pct", "yoy_change_pct", "cagr_pct"]);

export function metricMeta(key) {
  return METRIC_META[key] || { label: key || "-", unit: "", direction: "", source: "" };
}

export function metricLabel(key) {
  return metricMeta(key).label;
}

export function metricUnit(key) {
  return metricMeta(key).unit;
}

export function metricLabelWithUnit(key) {
  const meta = metricMeta(key);
  return meta.unit ? `${meta.label}（${meta.unit}）` : meta.label;
}

export function metricExplain(key) {
  const meta = metricMeta(key);
  const lines = [meta.label];
  if (meta.direction) lines.push(`判读：${meta.direction}`);
  if (meta.source) lines.push(meta.source);
  return lines.join("；");
}

export function metricSourceNote(key) {
  const meta = metricMeta(key);
  return meta.source || "";
}

export function formatMetricValue(key, value, formatters = {}) {
  const { fmt = (v, digits = 2) => Number(v).toFixed(digits), fmtPct = (v, digits = 2) => `${(Number(v) * 100).toFixed(digits)}%` } =
    formatters;
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";

  if (RATIO_PERCENT_KEYS.has(key)) {
    return fmtPct(number, 2);
  }
  if (VALUE_PERCENT_KEYS.has(key)) {
    return `${fmt(number, 2)}%`;
  }
  if (key === "runtime_ms") {
    return `${fmt(number, 2)} ms`;
  }
  if (key === "recommend_strength") {
    return `${fmt(number, 1)} / 100`;
  }
  return fmt(number, 2);
}
