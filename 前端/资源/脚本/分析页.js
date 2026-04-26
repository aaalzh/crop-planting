import { initProtectedPage, notify, refreshInteractions, setPageBusy } from "./公共.js";
import { fetchAssistantAnswer } from "./决策接口.js";
import { escapeHtml } from "./决策界面.js";

const DEFAULT_SESSION_TITLE = "新对话";
const initialFocusCrop = (() => {
  try {
    return new URLSearchParams(window.location.search).get("crop") || "";
  } catch (error) {
    return "";
  }
})();

const state = {
  sessions: [
    {
      id: "default",
      title: DEFAULT_SESSION_TITLE,
      history: []
    }
  ],
  active: "default",
  focusCrop: String(initialFocusCrop || "").trim()
};

function activeSession() {
  return state.sessions.find((session) => session.id === state.active) || state.sessions[0];
}

function buildSessionTitle(question) {
  const text = String(question || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return DEFAULT_SESSION_TITLE;
  }
  const match = text.match(/.+?(?:[。！？!?]|$)/u);
  const firstSentence = (match ? match[0] : text).replace(/[。！？!?]+$/u, "").trim();
  if (!firstSentence) {
    return DEFAULT_SESSION_TITLE;
  }
  return firstSentence.length > 24 ? `${firstSentence.slice(0, 24).trim()}...` : firstSentence;
}

function syncHeader() {
  const session = activeSession();
  const titleNode = document.getElementById("assistantTitle");
  if (titleNode) {
    titleNode.textContent = session?.title || DEFAULT_SESSION_TITLE;
  }
  const lastAssistant = [...(session?.history || [])].reverse().find((item) => item.role === "assistant");
  const sourceNode = document.getElementById("assistantSourceHint");
  if (sourceNode) {
    const parts = [];
    if (lastAssistant?.source?.label) {
      parts.push(`来源：${lastAssistant.source.label}`);
    }
    if (state.focusCrop) {
      parts.push(`聚焦作物：${state.focusCrop}`);
    }
    sourceNode.textContent = parts.join(" · ");
  }
}

function renderSessions() {
  const container = document.getElementById("assistantSessions");
  if (!container) return;

  container.innerHTML = state.sessions
    .map((session) => {
      return `
        <button class="session-card ${session.id === state.active ? "is-active" : ""}" data-session="${session.id}" type="button">
          <p class="session-title">${escapeHtml(session.title)}</p>
        </button>
      `;
    })
    .join("");

  container.querySelectorAll(".session-card").forEach((node) => {
    node.addEventListener("click", () => {
      state.active = node.dataset.session;
      renderSessions();
      renderHistory();
    });
  });
  syncHeader();
}

function renderHistory() {
  const container = document.getElementById("assistantHistory");
  if (!container) return;
  const history = activeSession().history;
  if (!history.length) {
    container.innerHTML =
      '<div class="chat-row"><div class="chat-bubble">输入一个问题即可开始对话。</div></div>';
    syncHeader();
    return;
  }
  container.innerHTML = history
    .map((item) => {
      if (item.role === "user") {
        return `
          <div class="chat-row user">
            <div class="chat-bubble" style="white-space:pre-wrap;">${escapeHtml(item.text || "")}</div>
          </div>
        `;
      }
      const bullets = (item.bullets || []).map((line) => `<li style="margin:4px 0;">${escapeHtml(line)}</li>`).join("");
      const sourceLine = item.source?.label ? `<p class="chat-meta">来源：${escapeHtml(item.source.label)}</p>` : "";
      return `
        <div class="chat-row assistant">
          <div>
            <div class="chat-bubble">
              <div style="white-space:pre-wrap;">${escapeHtml(item.text || "")}</div>
              ${bullets ? `<ul class="text-list">${bullets}</ul>` : ""}
            </div>
            ${sourceLine}
          </div>
        </div>
      `;
    })
    .join("");
  container.scrollTop = container.scrollHeight;
  syncHeader();
}

async function askQuestion() {
  const textarea = document.getElementById("assistantCustomQuestion");
  const question = (textarea?.value || "").trim();
  if (!question) {
    notify("请输入问题再发送。", "info");
    return;
  }

  textarea.value = "";
  const session = activeSession();
  const isFirstQuestion = !session.history.some((item) => item.role === "user");
  if (isFirstQuestion) {
    session.title = buildSessionTitle(question);
  }
  session.history.push({ role: "user", text: question });
  renderSessions();
  renderHistory();
  setPageBusy(true);

  try {
    const answer = await fetchAssistantAnswer(null, state.focusCrop || null, question);
    session.history.push({
      role: "assistant",
      text: answer.answer || "暂时无法回答。",
      bullets: answer.bullets || [],
      source: answer.source || {}
    });
    renderHistory();
  } catch (error) {
    notify(error.message || "AI 调用失败", "error");
  } finally {
    setPageBusy(false);
  }
}

function addSession() {
  const id = `session-${Date.now()}`;
  state.sessions.unshift({
    id,
    title: DEFAULT_SESSION_TITLE,
    history: []
  });
  state.active = id;
  renderSessions();
  renderHistory();
}

async function main() {
  await initProtectedPage();
  renderSessions();
  renderHistory();
  document.getElementById("assistantAskBtn")?.addEventListener("click", askQuestion);
  document.getElementById("assistantNewBtn")?.addEventListener("click", addSession);
  refreshInteractions(document);
}

main();
