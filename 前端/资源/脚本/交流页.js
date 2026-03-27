import { initProtectedPage, notify, refreshInteractions } from "./公共.js";
import { fetchCommunitySummary } from "./决策接口.js";
import { cardEmpty, escapeHtml } from "./决策界面.js";

const state = {
  threads: [],
  active: null
};

function threadById(id) {
  return state.threads.find((item) => item.id === id) || null;
}

function activeThread() {
  return threadById(state.active) || state.threads[0] || null;
}

function parseHash() {
  const match = String(window.location.hash || "").match(/^#thread-(.+)$/);
  return match ? decodeURIComponent(match[1]) : "";
}

function updateHash(id) {
  const next = `#thread-${encodeURIComponent(id)}`;
  if (window.location.hash !== next) {
    window.history.replaceState(null, "", next);
  }
}

function setActive(id, updateUrl = true) {
  if (!threadById(id)) return;
  state.active = id;
  if (updateUrl) updateHash(id);
  renderThreadList();
  renderReader();
}

function renderThreadList() {
  const container = document.getElementById("communityThreadList");
  const meta = document.getElementById("communityThreadMeta");
  if (!container || !meta) return;

  meta.textContent = `${state.threads.length} 条帖子`;
  container.innerHTML = state.threads.length
    ? state.threads
        .map(
          (item) => `
            <button class="community-thread-card ${item.id === state.active ? "is-active" : ""}" type="button" data-thread-id="${escapeHtml(item.id)}">
              <div class="community-thread-thumb">
                <img src="${escapeHtml(item.cover_image || "")}" alt="${escapeHtml(item.cover_alt || item.title || "帖子封面")}" loading="lazy">
              </div>
              <div class="community-thread-copy">
                <div class="stack-head">
                  <strong>${escapeHtml(item.title || "-")}</strong>
                  <span class="tone-pill tone-muted">${escapeHtml(item.tag || "论坛")}</span>
                </div>
                <p>${escapeHtml(item.summary || "")}</p>
                <div class="community-meta">
                  <span>${escapeHtml(item.author || "匿名农友")}</span>
                  <span>${escapeHtml(item.stats || "持续讨论中")}</span>
                </div>
              </div>
            </button>
          `
        )
        .join("")
    : cardEmpty("暂时没有交流内容。");

  container.querySelectorAll("[data-thread-id]").forEach((node) => {
    node.addEventListener("click", () => setActive(node.dataset.threadId));
  });
}

function renderReader() {
  const container = document.getElementById("communityReader");
  const thread = activeThread();
  if (!container) return;
  if (!thread) {
    container.innerHTML = cardEmpty("暂时没有可阅读的帖子。");
    return;
  }

  const body = (thread.body || []).map((row) => `<p>${escapeHtml(row)}</p>`).join("");
  const points = (thread.key_points || []).filter(Boolean).map((row) => `<li>${escapeHtml(row)}</li>`).join("");
  const comments = (thread.comments || [])
    .map(
      (row) => `
        <div class="community-comment-item">
          <div class="community-comment-head">
            <strong>${escapeHtml(row.author || "匿名农友")}</strong>
            <span>${escapeHtml(row.time || "")}</span>
          </div>
          <p>${escapeHtml(row.text || "")}</p>
        </div>
      `
    )
    .join("");

  container.innerHTML = `
    <div class="community-reader-hero">
      <img src="${escapeHtml(thread.cover_image || "")}" alt="${escapeHtml(thread.cover_alt || thread.title || "帖子封面")}" loading="lazy">
    </div>
    <div class="community-reader-body">
      <p class="eyebrow">${escapeHtml(thread.tag || "经验交流")}</p>
      <h2>${escapeHtml(thread.title || "")}</h2>
      <div class="community-reader-meta-row">
        <span>${escapeHtml(thread.author || "匿名农友")}</span>
        <span>${escapeHtml(thread.date || "")}</span>
        <span>${escapeHtml(thread.location || "")}</span>
        <span>${escapeHtml(thread.read_time || "")}</span>
        <span>${escapeHtml(thread.stats || "")}</span>
      </div>
      <p class="community-reader-lead">${escapeHtml(thread.summary || "")}</p>
      <div class="community-reader-section">${body}</div>
      <div class="community-reader-highlight">
        <h3>帖子要点</h3>
        <ul class="text-list">${points}</ul>
      </div>
      <div class="community-reader-highlight">
        <h3>评论区</h3>
        <div class="community-comment-list">${comments}</div>
      </div>
      <div class="community-image-credit">
        <span>封面署名：${escapeHtml(thread.cover_credit || "可再用素材")}</span>
        <div class="store-source-links">
          <a class="text-link" href="${escapeHtml(thread.cover_source_url || "#")}" target="_blank" rel="noreferrer">图片来源</a>
          <a class="text-link" href="${escapeHtml(thread.cover_license_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(thread.cover_license_label || "许可信息")}</a>
        </div>
      </div>
    </div>
  `;
  refreshInteractions(container);
}

function renderAll() {
  renderThreadList();
  renderReader();
  refreshInteractions(document);
}

async function main() {
  await initProtectedPage();
  try {
    const data = await fetchCommunitySummary();
    state.threads = data.highlights || [];
    document.getElementById("communitySummaryTitle").textContent = data.headline || "经验交流";
    document.getElementById("communitySummaryNote").textContent = data.subheadline || "";
    const hashThread = parseHash();
    state.active = threadById(hashThread)?.id || state.threads[0]?.id || null;
    renderAll();

    window.addEventListener("hashchange", () => {
      const id = parseHash();
      if (threadById(id)) {
        state.active = id;
        renderThreadList();
        renderReader();
      }
    });
  } catch (error) {
    notify(error.message || "经验交流页面加载失败", "error");
  }
}

main();
