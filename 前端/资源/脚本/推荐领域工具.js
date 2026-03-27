import { cropName } from "./公共.js";
import { clamp, safeNumber, toNumber } from "./推荐图表工具.js";

export const INPUT_KEYS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];

export const CONFIDENCE_MAP = {
  high: "高",
  mid: "中",
  low: "低"
};

export const SCORE_VIEW_DESCRIPTIONS = {
  summary: "快速查看候选作物的推荐强度与收益，适合先做直观初筛。",
  decomp: "拆解单个候选作物的内部排序依据，观察每个维度的贡献与短板。",
  compare: "前N候选横向对比推荐强度、收益强度与风险安全度，便于做取舍。"
};

export const VISUAL_CHART_IDS = {
  priceDaily: "priceDailyVisualChart",
  priceForecastDetail: "priceForecastDetailChart",
  priceWindowSeasonal: "priceWindowSeasonalChart",
  yield: "yieldVisualChart",
  cost: "costVisualChart",
  profile: "profileVisualChart",
  decisionRisk: "decisionRiskChart",
  decisionConfidence: "decisionConfidenceChart",
  scoreDecomp: "scoreDecompChart",
  scoreCompare: "scoreCompareChart"
};

export function scoreToLevel(score) {
  if (score >= 85) return "优秀";
  if (score >= 70) return "良好";
  if (score >= 55) return "一般";
  return "偏弱";
}

export function evaluateInputHealth(env) {
  const n = safeNumber(env.N);
  const p = safeNumber(env.P);
  const k = safeNumber(env.K);
  const temperature = safeNumber(env.temperature);
  const humidity = safeNumber(env.humidity);
  const ph = safeNumber(env.ph);
  const rainfall = safeNumber(env.rainfall);

  const npkValues = [n, p, k].filter(Number.isFinite);
  const npkScore = (() => {
    if (npkValues.length < 3) return 0;
    const mean = (npkValues[0] + npkValues[1] + npkValues[2]) / 3;
    if (mean <= 0) return 0;
    const variance = npkValues.reduce((acc, cur) => acc + (cur - mean) ** 2, 0) / npkValues.length;
    const cv = Math.sqrt(variance) / mean;
    return clamp(100 - cv * 150, 0, 100);
  })();

  const climateScore = (() => {
    if (![temperature, humidity, rainfall].every(Number.isFinite)) return 0;
    const tempPenalty = Math.abs(temperature - 26) * 2.2;
    const humidityPenalty = Math.abs(humidity - 68) * 0.9;
    const rainfallPenalty = Math.abs(rainfall - 120) * 0.12;
    return clamp(100 - tempPenalty - humidityPenalty - rainfallPenalty, 0, 100);
  })();

  const soilScore = (() => {
    if (![ph, k].every(Number.isFinite)) return 0;
    const phPenalty = Math.abs(ph - 6.6) * 16;
    const kPenalty = Math.max(0, Math.abs(k - 40) - 40) * 0.8;
    return clamp(100 - phPenalty - kPenalty, 0, 100);
  })();

  const allValues = [n, p, k, temperature, humidity, ph, rainfall];
  const completenessScore = Math.round((allValues.filter(Number.isFinite).length / allValues.length) * 100);

  return {
    npkScore,
    climateScore,
    soilScore,
    completenessScore
  };
}

export function recommendStrengthLevel(score) {
  const val = toNumber(score);
  if (val >= 82) return "很强";
  if (val >= 68) return "较强";
  if (val >= 55) return "中等";
  return "谨慎";
}

export function recommendStrengthFromRow(row) {
  const existing = safeNumber(row?.recommend_strength);
  if (existing !== null) return clamp(existing, 0, 100);

  const envFit = probabilityToRadarPercent(row?.env_prob);
  const hasProbBest = safeNumber(row?.prob_best) !== null;
  const decisionConfidence = hasProbBest ? probabilityToRadarPercent(row?.prob_best) : clamp(Math.max(38, envFit * 0.76), 3, 97);
  const riskSafety = riskToSafetyPercent(row?.risk);
  const profitBuffer = marginToRadarPercent(rowMarginPct(row));
  const rankSignal = scoreToRadarPercent(row?.score);

  return clamp(envFit * 0.34 + decisionConfidence * 0.2 + riskSafety * 0.2 + profitBuffer * 0.14 + rankSignal * 0.12, 0, 100);
}

export function normalizeRows(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map((row) => {
    const recommendStrength = recommendStrengthFromRow(row);
    const level =
      (typeof row?.recommend_strength_level === "string" && row.recommend_strength_level) || recommendStrengthLevel(recommendStrength);
    return {
      ...row,
      recommend_strength: recommendStrength,
      recommend_strength_level: level
    };
  });
}

export function sortRows(rows, sortState) {
  const key = sortState.key;
  const direction = sortState.direction === "asc" ? 1 : -1;
  const copied = [...rows];
  copied.sort((a, b) => {
    if (key === "crop") {
      const left = String(cropName(a?.[key]));
      const right = String(cropName(b?.[key]));
      return left.localeCompare(right, "zh-CN") * direction;
    }
    return (toNumber(a?.[key]) - toNumber(b?.[key])) * direction;
  });
  return copied;
}

export function emptyVisualPayload() {
  return {
    crop: "",
    price: {
      history: [],
      ma30: [],
      ma90: [],
      forecast: [],
      actual: [],
      raw: { history: [], ma30: [], ma90: [], forecast: [], actual: [] },
      stats: {}
    },
    yield: { history: [], trend: [], actual: [], forecast: {}, stats: {} },
    cost: { history: [], trend: [], actual: [], forecast: {}, stats: {} },
    prediction: {},
    profile: {},
    warnings: [],
    time_index: [],
    time_meta: {},
    alignment_notes: []
  };
}

export function toDatePairs(rows) {
  if (!Array.isArray(rows)) return [];
  return rows
    .map((item) => {
      const date = String(item?.date || "");
      const value = safeNumber(item?.value);
      if (!date || value === null) return null;
      return [date, value];
    })
    .filter(Boolean);
}

export function buildYearSeries(rows, forecast) {
  const history = Array.isArray(rows)
    ? rows
        .map((item) => ({ year: String(item?.year ?? ""), value: safeNumber(item?.value) }))
        .filter((item) => item.year && item.value !== null)
    : [];
  const trend = Array.isArray(forecast)
    ? forecast
        .map((item) => ({ year: String(item?.year ?? ""), value: safeNumber(item?.value) }))
        .filter((item) => item.year && item.value !== null)
    : [];
  return { history, trend };
}

export function fitLinearPoint(rows, targetYear) {
  const samples = (rows || [])
    .map((item) => ({
      x: toNumber(item?.year),
      y: safeNumber(item?.value)
    }))
    .filter((item) => Number.isFinite(item.x) && item.y !== null);
  if (samples.length < 2) return null;

  const n = samples.length;
  const sumX = samples.reduce((acc, cur) => acc + cur.x, 0);
  const sumY = samples.reduce((acc, cur) => acc + cur.y, 0);
  const sumXX = samples.reduce((acc, cur) => acc + cur.x * cur.x, 0);
  const sumXY = samples.reduce((acc, cur) => acc + cur.x * cur.y, 0);
  const denom = n * sumXX - sumX * sumX;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-9) return null;

  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  const y = slope * toNumber(targetYear) + intercept;
  return safeNumber(y);
}

export function probabilityToRadarPercent(probability) {
  const p = clamp(toNumber(probability), 0, 1);
  const softened = 0.03 + p * 0.94;
  return clamp(softened * 100, 3, 97);
}

export function riskToSafetyPercent(riskScoreVal) {
  const r = Math.max(0, toNumber(riskScoreVal));
  const baseline = 100 / (1 + Math.pow(r / 0.2, 1.25));
  return clamp(baseline, 3, 97);
}

export function marginToRadarPercent(marginPctVal) {
  const m = toNumber(marginPctVal);
  const centered = (m - 10) / 18;
  const score = 50 + 42 * Math.tanh(centered);
  return clamp(score, 3, 97);
}

export function scoreToRadarPercent(scoreVal) {
  const s = toNumber(scoreVal);
  const scaled = 50 + 45 * Math.tanh(s / 18000);
  return clamp(scaled, 3, 97);
}

export function sampleStabilityPercent(afterCount, beforeCount) {
  const after = Math.max(0, toNumber(afterCount));
  const before = Math.max(after, toNumber(beforeCount));
  const coverage = before > 0 ? clamp(after / before, 0, 1) : 0;
  const countTerm = 1 - Math.exp(-after / 4);
  const score = (countTerm * 0.75 + Math.sqrt(coverage) * 0.25) * 100;
  return clamp(score, 3, 97);
}

export function rowMarginPct(row) {
  const price = safeNumber(row?.price_pred);
  const yld = safeNumber(row?.yield);
  const cost = safeNumber(row?.cost_pred);
  if (price === null || yld === null || cost === null) return null;
  const revenue = price * yld;
  if (!Number.isFinite(revenue) || revenue <= 0) return null;
  return safeNumber(((revenue - cost) / revenue) * 100);
}

export function profitStrengthPercent(profit, refAbs) {
  const base = Math.max(1, Number(refAbs) || 1);
  const ratio = toNumber(profit) / base;
  return clamp(50 + 45 * Math.tanh(ratio), 3, 97);
}

export function yieldStrengthPercent(yieldValue, refMax) {
  const base = Math.max(1e-9, Number(refMax) || 1);
  const ratio = toNumber(yieldValue) / base;
  return clamp(ratio * 100, 3, 97);
}

export function topCompareRows(rows, limit = 6) {
  if (!rows.length) return [];
  return [...rows]
    .sort((a, b) => toNumber(b?.score) - toNumber(a?.score))
    .slice(0, Math.max(1, limit));
}
