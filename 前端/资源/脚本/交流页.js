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
  return threadById(state.active) || null;
}

function parseHash() {
  const match = String(window.location.hash || "").match(/^#thread-(.+)$/);
  return match ? decodeURIComponent(match[1]) : "";
}

function updateHash(id) {
  const nextHash = id ? `#thread-${encodeURIComponent(id)}` : "";
  if (window.location.hash === nextHash) return;
  const nextUrl = `${window.location.pathname}${window.location.search}${nextHash}`;
  window.history.replaceState(null, "", nextUrl);
}

function setActive(id, updateUrl = true) {
  if (!threadById(id)) return;
  state.active = id;
  if (updateUrl) updateHash(id);
  renderAll();
}

function clearActive(updateUrl = true) {
  state.active = null;
  if (updateUrl) updateHash("");
  renderAll();
}

function renderLayout() {
  const shell = document.getElementById("communityShell");
  const feedCard = document.getElementById("communityFeedCard");
  const readerCard = document.getElementById("communityReader");
  const reading = Boolean(activeThread());

  shell?.classList.toggle("is-reading", reading);
  feedCard?.classList.toggle("is-hidden", reading);
  readerCard?.classList.toggle("is-hidden", !reading);
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
                <div class="community-thread-topline">
                  <span class="tone-pill tone-muted">${escapeHtml(item.tag || "论坛")}</span>
                  ${item.date ? `<span>${escapeHtml(item.date)}</span>` : ""}
                </div>
                <div class="stack-head">
                  <strong>${escapeHtml(item.title || "-")}</strong>
                </div>
                <p>${escapeHtml(item.summary || "")}</p>
                <div class="community-meta">
                  <span>${escapeHtml(item.author || "匿名农友")}</span>
                  ${item.location ? `<span>${escapeHtml(item.location)}</span>` : ""}
                  <span>${escapeHtml(item.stats || "持续讨论中")}</span>
                </div>
                <span class="community-thread-cta">点击查看完整帖子</span>
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
    container.innerHTML = "";
    return;
  }

  const body = (thread.body || []).map((row) => `<p>${escapeHtml(row)}</p>`).join("");
  const pointRows = (thread.key_points || []).filter(Boolean);
  const commentRows = Array.isArray(thread.comments) ? thread.comments : [];
  const points = pointRows.length
    ? `<ul class="text-list">${pointRows.map((row) => `<li>${escapeHtml(row)}</li>`).join("")}</ul>`
    : `<p class="muted">暂无帖子要点。</p>`;
  const comments = commentRows.length
    ? commentRows
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
        .join("")
    : `<p class="muted">暂无评论。</p>`;

  container.innerHTML = `
    <div class="community-reader-toolbar">
      <button id="communityBackBtn" class="btn btn-secondary" type="button">返回帖子列表</button>
      <span class="muted">完整帖子阅读</span>
    </div>
    <div class="community-reader-hero">
      <img src="${escapeHtml(thread.cover_image || "")}" alt="${escapeHtml(thread.cover_alt || thread.title || "帖子封面")}" loading="lazy">
    </div>
    <div class="community-reader-body">
      <p class="eyebrow">${escapeHtml(thread.tag || "经验交流")}</p>
      <h2>${escapeHtml(thread.title || "")}</h2>
      <div class="community-reader-meta-row">
        <span>${escapeHtml(thread.author || "匿名农友")}</span>
        ${thread.date ? `<span>${escapeHtml(thread.date)}</span>` : ""}
        ${thread.location ? `<span>${escapeHtml(thread.location)}</span>` : ""}
        ${thread.read_time ? `<span>${escapeHtml(thread.read_time)}</span>` : ""}
        ${thread.stats ? `<span>${escapeHtml(thread.stats)}</span>` : ""}
      </div>
      <p class="community-reader-lead">${escapeHtml(thread.summary || "")}</p>
      <div class="community-reader-section">${body}</div>
      <div class="community-reader-highlight">
        <h3>帖子要点</h3>
        ${points}
      </div>
      <div class="community-reader-highlight">
        <h3>评论区</h3>
        <div class="community-comment-list">${comments}</div>
      </div>
    </div>
  `;
  container.querySelector("#communityBackBtn")?.addEventListener("click", () => clearActive());
  refreshInteractions(container);
}

function renderAll() {
  renderThreadList();
  renderReader();
  renderLayout();
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
    state.active = threadById(hashThread)?.id || null;
    renderAll();

    window.addEventListener("hashchange", () => {
      const id = parseHash();
      state.active = threadById(id)?.id || null;
      renderAll();
    });
  } catch (error) {
    notify(error.message || "经验交流页面加载失败", "error");
  }
}

main();
