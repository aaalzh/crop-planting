function isReduceMotion() {
  const mediaReduced =
    typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const classReduced = document.body?.classList?.contains("reduce-motion");
  return Boolean(mediaReduced || classReduced);
}

export function clamp(num, min, max) {
  return Math.max(min, Math.min(max, num));
}

export function safeNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export function toNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
}

export function hasEcharts() {
  return Boolean(window.echarts && typeof window.echarts.init === "function");
}

export function chartAnimationBase(extraDelay = 0) {
  if (isReduceMotion()) {
    return { animation: false };
  }
  return {
    animation: true,
    animationDuration: 1500,
    animationEasing: "quarticOut",
    animationDurationUpdate: 780,
    animationEasingUpdate: "cubicInOut",
    animationDelay: extraDelay,
    animationDelayUpdate: Math.floor(extraDelay / 2)
  };
}

export function seriesMotion(seriesIndex, pointStep = 8, pointCap = 42) {
  if (isReduceMotion()) {
    return { animation: false };
  }
  const baseDelay = seriesIndex * 180;
  const step = Math.max(0, pointStep);
  const cap = Math.max(0, pointCap);
  return {
    animation: true,
    animationDuration: 1600,
    animationEasing: "quarticOut",
    animationDelay: (idx) => baseDelay + Math.min(Number(idx) || 0, cap) * step,
    animationDurationUpdate: 820,
    animationEasingUpdate: "cubicOut",
    animationDelayUpdate: (idx) => Math.floor(baseDelay / 2) + Math.min(Number(idx) || 0, cap) * Math.max(2, Math.floor(step / 2))
  };
}

