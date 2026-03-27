import { fmt } from "./公共.js";
import { chartAnimationBase, seriesMotion } from "./推荐图表工具.js";

export function buildScoreDecompOption(decompItems) {
  return {
    ...chartAnimationBase(180),
    tooltip: {
      trigger: "item",
      show: true,
      triggerOn: "mousemove|click",
      appendToBody: true,
      confine: false,
      formatter: (params) => {
        const p = Array.isArray(params) ? params[0] : params;
        const item = decompItems[p?.dataIndex ?? 0];
        if (!item) return "";
        return [`<strong>${item.name}</strong>`, `标准化得分：${fmt(item.value, 1)} / 100`, `原值：${item.raw}`, item.explain].join("<br/>");
      }
    },
    grid: { left: 96, right: 24, top: 24, bottom: 24, containLabel: true },
    xAxis: { type: "value", max: 100, name: "标准化得分（0-100）" },
    yAxis: {
      type: "category",
      data: decompItems.map((item) => item.name),
      axisLabel: { width: 88, overflow: "truncate" }
    },
    series: [
      {
        ...seriesMotion(0, 0, 0),
        type: "bar",
        data: decompItems.map((item) => Number(item.value.toFixed(2))),
        barWidth: 16,
        itemStyle: { borderRadius: [0, 8, 8, 0], color: "#2f7b54" }
      }
    ]
  };
}

export function buildScoreCompareOption(compareRows, formatMetric) {
  return {
    ...chartAnimationBase(240),
    tooltip: {
      trigger: "item",
      show: true,
      triggerOn: "mousemove|click",
      appendToBody: true,
      confine: false,
      formatter: (params) => {
        const lines = [];
        const index = Array.isArray(params) && params.length ? params[0].dataIndex : 0;
        const pick = compareRows[index];
        if (!pick) return "";
        lines.push(`<strong>${pick.crop}</strong>`);
        lines.push(`综合分强度：${fmt(pick.scoreStrength, 1)} / 100（原值 ${formatMetric("score", pick.row?.score)}）`);
        lines.push(`收益强度：${fmt(pick.profitStrength, 1)} / 100（原值 ${formatMetric("profit", pick.row?.profit)}）`);
        lines.push(`风险安全度：${fmt(pick.riskSafety, 1)} / 100（原值 ${formatMetric("risk", pick.row?.risk)}）`);
        lines.push("判读：综合分与收益越大越好，风险分越小越好。");
        return lines.join("<br/>");
      }
    },
    legend: {
      type: "scroll",
      top: 0,
      left: 0,
      right: 0,
      data: ["综合分强度", "收益强度", "风险安全度"]
    },
    grid: { left: 48, right: 20, top: 56, bottom: 42, containLabel: true },
    xAxis: {
      type: "category",
      name: "候选作物",
      data: compareRows.map((item) => item.crop),
      axisLabel: {
        interval: 0,
        rotate: compareRows.length > 4 ? 20 : 0,
        hideOverlap: true
      }
    },
    yAxis: { type: "value", max: 100, name: "标准化得分（0-100）", axisLabel: { hideOverlap: true } },
    series: [
      {
        ...seriesMotion(0, 12, 28),
        name: "综合分强度",
        type: "bar",
        barMaxWidth: 20,
        data: compareRows.map((item) => Number(item.scoreStrength.toFixed(2)))
      },
      {
        ...seriesMotion(1, 12, 28),
        name: "收益强度",
        type: "bar",
        barMaxWidth: 20,
        data: compareRows.map((item) => Number(item.profitStrength.toFixed(2)))
      },
      {
        ...seriesMotion(2, 12, 28),
        name: "风险安全度",
        type: "bar",
        barMaxWidth: 20,
        data: compareRows.map((item) => Number(item.riskSafety.toFixed(2)))
      }
    ]
  };
}
