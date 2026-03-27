import { cropName, fmt } from "./公共.js";

export { cropName, fmt };

export function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function money(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `₹${Number(value).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  })}`;
}

export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(digits)}%`;
}

export function signedPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const num = Number(value);
  const prefix = num > 0 ? "+" : "";
  return `${prefix}${num.toFixed(digits)}%`;
}

export function riskText(level) {
  if (!level) return "未知";
  return level;
}

export function riskTone(level) {
  if (level === "高") return "danger";
  if (level === "中") return "warn";
  if (level === "低") return "safe";
  return "muted";
}

export function trendTone(direction) {
  if (direction === "走强") return "safe";
  if (direction === "走弱") return "danger";
  return "muted";
}

export function styleTone(style) {
  if (style === "更赚") return "warn";
  if (style === "更稳") return "safe";
  return "muted";
}

export function questionButtons(questions) {
  return (questions || [])
    .map(
      (item) => `
        <button type="button" class="quick-chip" data-question-id="${escapeHtml(item.id)}">
          ${escapeHtml(item.label)}
        </button>
      `
    )
    .join("");
}

export function listItems(items, className = "text-list") {
  const rows = (items || []).filter(Boolean);
  if (!rows.length) {
    return `<p class="muted">暂无内容</p>`;
  }
  return `<ul class="${className}">${rows.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

export function cardEmpty(text) {
  return `<p class="muted">${escapeHtml(text || "暂无内容")}</p>`;
}
