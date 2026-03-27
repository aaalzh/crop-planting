import { initProtectedPage, notify, refreshInteractions } from "./公共.js";
import { fetchProfileSummary } from "./决策接口.js";
import { cardEmpty, escapeHtml, money } from "./决策界面.js";

async function main() {
  await initProtectedPage();
  try {
    const data = await fetchProfileSummary();
    document.getElementById("profileHistoryList").innerHTML = (data.history || []).length
      ? data.history
          .map((item) => {
            const best = item.best_crop || {};
            return `
              <article class="preview-card">
                <div class="stack-head">
                  <strong>${escapeHtml(best.crop_label || "-")}</strong>
                  <span class="muted">${escapeHtml(item.created_at || "")}</span>
                </div>
                <p>预计收益 ${money(best.profit)} · 风险 ${best.risk === null || best.risk === undefined ? "-" : Number(best.risk).toFixed(2)}</p>
              </article>
            `;
          })
          .join("")
      : cardEmpty("还没有历史推荐记录。");
    document.getElementById("profileFavoritesList").innerHTML = (data.favorites || []).length
      ? data.favorites
          .map(
            (item) => `
              <article class="preview-card">
                <strong>${escapeHtml(item.crop_label || "-")}</strong>
                <p>最近被选为主推 ${escapeHtml(item.count)} 次</p>
              </article>
            `
          )
          .join("")
      : cardEmpty("当前还没有形成偏好作物。");
    document.getElementById("profileSettingsList").innerHTML = (data.settings || []).length
      ? data.settings
          .map(
            (item) => `
              <article class="preview-card">
                <strong>${escapeHtml(item.label || "")}</strong>
                <p>${escapeHtml(item.value || "")}</p>
              </article>
            `
          )
          .join("")
      : cardEmpty("暂无设置项。");
    refreshInteractions(document);
  } catch (error) {
    notify(error.message || "个人页加载失败", "error");
  }
}

main();
