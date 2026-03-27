export const CROP_MAP = {
  apple: "苹果",
  banana: "香蕉",
  blackgram: "黑豆",
  chickpea: "鹰嘴豆",
  coconut: "椰子",
  cocount: "椰子",
  coffee: "咖啡",
  cotton: "棉花",
  grapes: "葡萄",
  jute: "黄麻",
  kidneybeans: "芸豆",
  lentil: "扁豆",
  maize: "玉米",
  mango: "芒果",
  mothbeans: "木豆",
  mungbean: "绿豆",
  muskmelon: "香瓜",
  orange: "橙子",
  papaya: "木瓜",
  pigeonpeas: "豌豆",
  pomegranate: "石榴",
  rice: "水稻",
  watermelon: "西瓜",
  "water melon": "西瓜"
};

export const $ = (id) => document.getElementById(id);
const LOGIN_PAGE_URL = "/";
const STORAGE_KEYS = {
  reduceMotion: "crop_ui_reduce_motion"
};

const state = {
  busyCount: 0,
  shellReady: false,
  paletteOpen: false,
  paletteItems: [],
  paletteView: {
    filtered: [],
    activeIndex: 0
  }
};

const UI_IDS = {
  loadingBar: "globalLoadingBar",
  toastStack: "globalToastStack",
  backTop: "globalBackTop",
  commandPalette: "globalCommandPalette",
  commandInput: "globalCommandInput",
  commandList: "globalCommandList"
};

let revealObserver = null;

function isInputLike(node) {
  if (!node || !(node instanceof HTMLElement)) return false;
  const tag = node.tagName.toLowerCase();
  return (
    tag === "input" ||
    tag === "textarea" ||
    tag === "select" ||
    node.isContentEditable ||
    Boolean(node.closest("[contenteditable='true']"))
  );
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function ensureLoadingBar() {
  let node = $(UI_IDS.loadingBar);
  if (node) return node;
  node = document.createElement("div");
  node.id = UI_IDS.loadingBar;
  node.className = "loading-bar";
  node.innerHTML = `<span class="loading-bar-fill"></span>`;
  document.body.prepend(node);
  return node;
}

function ensureToastStack() {
  let node = $(UI_IDS.toastStack);
  if (node) return node;
  node = document.createElement("section");
  node.id = UI_IDS.toastStack;
  node.className = "toast-stack";
  node.setAttribute("aria-live", "polite");
  node.setAttribute("aria-atomic", "false");
  document.body.appendChild(node);
  return node;
}

function ensureBackTopButton() {
  let node = $(UI_IDS.backTop);
  if (node) return node;
  node = document.createElement("button");
  node.id = UI_IDS.backTop;
  node.type = "button";
  node.className = "backtop-btn";
  node.textContent = "回到顶部";
  node.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  document.body.appendChild(node);
  return node;
}

function updateBackTopState() {
  const node = ensureBackTopButton();
  node.classList.toggle("visible", window.scrollY > 360);
}

function applyStoredPrefs() {
  const reduceMotion = localStorage.getItem(STORAGE_KEYS.reduceMotion) === "1";
  document.body.classList.toggle("reduce-motion", reduceMotion);
}

function setReduceMotion(next) {
  localStorage.setItem(STORAGE_KEYS.reduceMotion, next ? "1" : "0");
  document.body.classList.toggle("reduce-motion", Boolean(next));
}

function tableRowInteractive(row) {
  if (!(row instanceof HTMLElement) || row.dataset.enhanced === "1") return;
  row.dataset.enhanced = "1";
  row.tabIndex = 0;
  row.classList.add("row-interactive");

  const selectRow = () => {
    row
      .closest("tbody")
      ?.querySelectorAll("tr.is-selected")
      .forEach((item) => item.classList.remove("is-selected"));
    row.classList.add("is-selected");
  };

  row.addEventListener("click", selectRow);
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectRow();
    }
  });
}

function getRevealObserver() {
  if (revealObserver || typeof IntersectionObserver === "undefined") {
    return revealObserver;
  }
  revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("reveal-in");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      rootMargin: "0px 0px -8% 0px",
      threshold: 0.18
    }
  );
  return revealObserver;
}

function buildCommandItems() {
  const navItems = Array.from(document.querySelectorAll(".nav-link"))
    .filter((node) => !node.hidden)
    .map((node) => {
      const href = node.getAttribute("href") || "#";
      const label = node.textContent?.trim() || href;
      return {
        title: `前往 ${label}`,
        hint: href,
        keywords: [label, href, "跳转"],
        run: () => window.location.assign(href)
      };
    });

  return [
    ...navItems,
    {
      title: "刷新当前页面",
      hint: "Ctrl/Cmd + R",
      keywords: ["刷新", "reload"],
      run: () => window.location.reload()
    },
    {
      title: "回到顶部",
      hint: "滚动定位",
      keywords: ["顶部", "scroll", "上方"],
      run: () => window.scrollTo({ top: 0, behavior: "smooth" })
    },
    {
      title: "切换动效模式",
      hint: document.body.classList.contains("reduce-motion") ? "当前: 低动效" : "当前: 标准动效",
      keywords: ["动效", "reduced motion", "动画"],
      run: () => {
        const next = !document.body.classList.contains("reduce-motion");
        setReduceMotion(next);
        notify(next ? "已切换为低动效模式" : "已恢复标准动效", "success");
        state.paletteItems = buildCommandItems();
        renderCommandList();
      }
    },
    {
      title: "退出登录",
      hint: "安全退出会话",
      keywords: ["logout", "退出"],
      run: () => logout()
    }
  ];
}

function runCommandByIndex(idx) {
  const picked = state.paletteView.filtered[idx];
  if (!picked) return;
  closeCommandPalette();
  picked.run?.();
}

function renderCommandList() {
  const input = $(UI_IDS.commandInput);
  const listNode = $(UI_IDS.commandList);
  if (!listNode) return;

  const keyword = (input?.value || "").trim().toLowerCase();
  state.paletteView.filtered = state.paletteItems.filter((item) => {
    if (!keyword) return true;
    const haystack = [item.title, item.hint, ...(item.keywords || [])].join(" ").toLowerCase();
    return haystack.includes(keyword);
  });

  if (!state.paletteView.filtered.length) {
    state.paletteView.activeIndex = 0;
    listNode.innerHTML = `<div class="command-empty">没有匹配项，请换个关键词。</div>`;
    return;
  }

  state.paletteView.activeIndex = Math.min(state.paletteView.activeIndex, state.paletteView.filtered.length - 1);
  listNode.innerHTML = state.paletteView.filtered
    .slice(0, 12)
    .map((item, idx) => {
      const activeClass = idx === state.paletteView.activeIndex ? "active" : "";
      return `
        <button type="button" class="command-item ${activeClass}" data-command-index="${idx}">
          <strong>${escapeHtml(item.title)}</strong>
          <span>${escapeHtml(item.hint || "")}</span>
        </button>
      `;
    })
    .join("");

  listNode.querySelectorAll("[data-command-index]").forEach((node) => {
    node.addEventListener("click", () => {
      const idx = Number(node.getAttribute("data-command-index"));
      runCommandByIndex(idx);
    });
  });
}

function moveCommandCursor(delta) {
  if (!state.paletteView.filtered.length) return;
  const len = Math.min(12, state.paletteView.filtered.length);
  state.paletteView.activeIndex = (state.paletteView.activeIndex + delta + len) % len;
  renderCommandList();
}

function handlePaletteKeydown(event) {
  if (event.key === "Escape") {
    event.preventDefault();
    closeCommandPalette();
    return;
  }
  if (event.key === "ArrowDown") {
    event.preventDefault();
    moveCommandCursor(1);
    return;
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    moveCommandCursor(-1);
    return;
  }
  if (event.key === "Enter") {
    event.preventDefault();
    runCommandByIndex(state.paletteView.activeIndex);
  }
}

function ensureCommandPalette() {
  let node = $(UI_IDS.commandPalette);
  if (node) return node;

  node = document.createElement("aside");
  node.id = UI_IDS.commandPalette;
  node.className = "command-palette";
  node.hidden = true;
  node.innerHTML = `
    <div class="command-mask" data-command-close></div>
    <section class="command-box" role="dialog" aria-modal="true" aria-label="快捷命令面板">
      <div class="command-head">
        <input id="${UI_IDS.commandInput}" type="search" placeholder="输入关键字，例如：推荐 / AI / 退出">
        <button type="button" class="btn btn-secondary command-close" data-command-close>关闭</button>
      </div>
      <p class="muted command-hint">Enter 执行 · ↑↓ 选择 · Esc 关闭</p>
      <div id="${UI_IDS.commandList}" class="command-list"></div>
    </section>
  `;
  document.body.appendChild(node);

  node.querySelectorAll("[data-command-close]").forEach((item) => {
    item.addEventListener("click", () => closeCommandPalette());
  });
  node.addEventListener("keydown", (event) => handlePaletteKeydown(event));
  $(UI_IDS.commandInput)?.addEventListener("input", () => renderCommandList());

  return node;
}

function openCommandPalette() {
  const palette = ensureCommandPalette();
  state.paletteItems = buildCommandItems();
  state.paletteOpen = true;
  state.paletteView.activeIndex = 0;
  palette.hidden = false;
  document.body.classList.add("command-open");
  renderCommandList();
  $(UI_IDS.commandInput)?.focus();
}

function closeCommandPalette() {
  const palette = ensureCommandPalette();
  state.paletteOpen = false;
  palette.hidden = true;
  document.body.classList.remove("command-open");
}

function toggleCommandPalette() {
  if (state.paletteOpen) {
    closeCommandPalette();
  } else {
    openCommandPalette();
  }
}

function bindShortcuts() {
  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === "k") {
      event.preventDefault();
      toggleCommandPalette();
      return;
    }

    if (state.paletteOpen) {
      handlePaletteKeydown(event);
      return;
    }

    if (!isInputLike(event.target) && event.key === "/") {
      event.preventDefault();
      openCommandPalette();
      return;
    }

    if (!event.altKey || event.ctrlKey || event.metaKey) return;
    const quickNav = {
      "1": "/home",
      "2": "/recommend",
      "3": "/assistant",
      "4": "/store",
      "5": "/community",
      "6": "/profile"
    };
    const url = quickNav[event.key];
    if (!url) return;

    const navNode = Array.from(document.querySelectorAll(".nav-link")).find(
      (node) => !node.hidden && (node.getAttribute("href") || "") === url
    );
    if (navNode) {
      event.preventDefault();
      window.location.assign(url);
    }
  });
}

function initShell() {
  ensureLoadingBar();
  ensureToastStack();
  ensureBackTopButton();
  ensureCommandPalette();
  applyStoredPrefs();
  updateBackTopState();

  if (!state.shellReady) {
    bindShortcuts();
    window.addEventListener("scroll", updateBackTopState, { passive: true });
    state.shellReady = true;
  }
}

export function refreshInteractions(root = document) {
  root.querySelectorAll(".table tbody tr").forEach((row) => tableRowInteractive(row));

  const revealTargets = root.querySelectorAll(
    ".hero, .card, .kpi, .bar-item, .insight-chip, .summary-pill-card, .gallery-group, .gallery-card, .table-wrap, .auth-card, .auth-hero, .feature-card, .decision-card, .preview-card, .stack-item, .mini-card, .conversation-item, .quick-chip"
  );
  const observer = getRevealObserver();
  revealTargets.forEach((node, idx) => {
    if (!(node instanceof HTMLElement) || node.dataset.revealReady === "1") return;
    node.dataset.revealReady = "1";
    node.classList.add("reveal-target");
    node.style.setProperty("--reveal-delay", `${Math.min(idx * 26, 240)}ms`);
    if (observer) {
      observer.observe(node);
    } else {
      node.classList.add("reveal-in");
    }
  });
}

export function cropName(v) {
  return CROP_MAP[v] || v || "-";
}

export function fmt(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toLocaleString("zh-CN", { maximumFractionDigits: digits });
}

export function fmtPct(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

export function setPageBusy(loading) {
  if (loading) {
    state.busyCount += 1;
  } else {
    state.busyCount = Math.max(0, state.busyCount - 1);
  }

  const active = state.busyCount > 0;
  const bar = ensureLoadingBar();
  bar.classList.toggle("active", active);
  if (!active) {
    bar.classList.add("done");
    window.setTimeout(() => bar.classList.remove("done"), 360);
  }
  document.body.classList.toggle("is-busy", active);
}

export function notify(message, type = "info", durationMs = 2800) {
  if (!message) return;
  const stack = ensureToastStack();
  const icons = {
    info: "i",
    success: "✓",
    error: "!"
  };
  const tone = ["info", "success", "error"].includes(type) ? type : "info";
  const toast = document.createElement("article");
  toast.className = `toast toast-${tone}`;
  toast.setAttribute("role", tone === "error" ? "alert" : "status");
  toast.innerHTML = `
    <div class="toast-icon">${icons[tone]}</div>
    <div class="toast-text">${escapeHtml(message)}</div>
    <button type="button" class="toast-close" aria-label="关闭通知">x</button>
  `;
  stack.appendChild(toast);

  const remove = () => {
    if (!toast.isConnected) return;
    toast.classList.add("closing");
    window.setTimeout(() => toast.remove(), 220);
  };

  toast.querySelector(".toast-close")?.addEventListener("click", remove);
  window.setTimeout(remove, Math.max(1400, Number(durationMs) || 2800));
}

export async function apiFetch(url, options = {}) {
  const opts = {
    credentials: "include",
    ...options
  };

  if (opts.body && typeof opts.body !== "string") {
    opts.headers = {
      "Content-Type": "application/json",
      ...(opts.headers || {})
    };
    opts.body = JSON.stringify(opts.body);
  }

  setPageBusy(true);
  try {
    const res = await fetch(url, opts);

    let payload = null;
    const text = await res.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_) {
        payload = null;
      }
    }

    if (!res.ok) {
      const msg = payload?.detail || `${res.status} ${res.statusText}`;
      if (res.status === 401 && !url.startsWith("/api/auth/")) {
        window.location.href = LOGIN_PAGE_URL;
      }
      throw new Error(msg);
    }
    return payload;
  } finally {
    setPageBusy(false);
  }
}

export async function requireUser() {
  const data = await apiFetch("/api/auth/me");
  return data.user;
}

export function setStatus(node, text, isError = false) {
  if (!node) return;
  node.textContent = text || "";
  node.classList.toggle("error", Boolean(isError));
  node.classList.toggle("ok", Boolean(text) && !isError);
  node.setAttribute("role", isError ? "alert" : "status");
  if (isError && text) {
    notify(text, "error", 4200);
  }
}

export async function logout() {
  try {
    await apiFetch("/api/auth/logout", { method: "POST" });
  } catch (_) {
    // fallback to server-side logout route
  } finally {
    window.location.assign("/logout");
  }
}

export function markActiveNav() {
  const current = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach((node) => {
    const href = node.getAttribute("href") || "";
    const active = href === current;
    node.classList.toggle("active", active);
    if (active) {
      node.setAttribute("aria-current", "page");
    } else {
      node.removeAttribute("aria-current");
    }
  });
}

export function applyUserRoleUI(user) {
  const isAdmin = String(user?.role || "").toLowerCase() === "admin";
  document.querySelectorAll("[data-admin-only]").forEach((node) => {
    node.hidden = !isAdmin;
  });
}

export function wireLogoutButton(buttonId = "logoutBtn") {
  const btn = $(buttonId);
  if (!btn) return;
  btn.addEventListener("click", () => {
    logout();
  });
}

export function initPublicPage() {
  initShell();
  refreshInteractions(document);
}

export async function initProtectedPage(userNameId = "userName") {
  const user = await requireUser();
  initShell();
  const userNode = $(userNameId);
  if (userNode) {
    const roleText = user.role === "admin" ? "管理员" : "普通用户";
    const stateText = user.enabled ? "启用" : "禁用";
    userNode.textContent = `${user.display_name} (${user.username}) · ${roleText} · ${stateText}`;
  }
  applyUserRoleUI(user);
  markActiveNav();
  wireLogoutButton();
  refreshInteractions(document);
  return user;
}
