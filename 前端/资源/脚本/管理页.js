import { $, apiFetch, initProtectedPage, refreshInteractions, setStatus } from "./公共.js";

const adminStats = $("adminStats");
const adminUsersBody = $("adminUsersBody");
const adminStatus = $("adminStatus");

const state = {
  currentUser: null,
  users: []
};

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderStats(stats) {
  const cards = [
    { label: "总账号数", value: stats.total },
    { label: "管理员", value: stats.admins },
    { label: "普通用户", value: stats.normal_users },
    { label: "启用账号", value: stats.enabled },
    { label: "禁用账号", value: stats.disabled }
  ];

  adminStats.innerHTML = cards
    .map(
      (c) => `
        <article class="kpi">
          <div class="label">${c.label}</div>
          <div class="value">${c.value ?? "-"}</div>
        </article>
      `
    )
    .join("");
}

function userRow(user) {
  const isSelf = String(user.username) === String(state.currentUser?.username);
  return `
    <tr data-username="${escapeHtml(user.username)}">
      <td>${escapeHtml(user.username)}</td>
      <td>
        <input type="text" data-field="display_name" maxlength="32" value="${escapeHtml(user.display_name || user.username)}">
      </td>
      <td>
        <select data-field="role" ${isSelf ? "disabled" : ""}>
          <option value="admin" ${user.role === "admin" ? "selected" : ""}>admin</option>
          <option value="user" ${user.role === "user" ? "selected" : ""}>user</option>
        </select>
      </td>
      <td>
        <select data-field="enabled" ${isSelf ? "disabled" : ""}>
          <option value="true" ${user.enabled ? "selected" : ""}>启用</option>
          <option value="false" ${user.enabled ? "" : "selected"}>禁用</option>
        </select>
      </td>
      <td>${escapeHtml(user.created_at || "-")}</td>
      <td>${escapeHtml(user.last_login_at || "-")}</td>
      <td>
        <button type="button" class="btn btn-secondary" data-action="save">保存</button>
      </td>
    </tr>
  `;
}

function renderUsers(users) {
  if (!users.length) {
    adminUsersBody.innerHTML = `<tr><td colspan="7">暂无用户</td></tr>`;
    return;
  }
  adminUsersBody.innerHTML = users.map(userRow).join("");
  refreshInteractions(document);
}

async function loadUsers() {
  const data = await apiFetch("/api/admin/users");
  state.users = data.users || [];
  renderStats(data.stats || {});
  renderUsers(state.users);
}

adminUsersBody.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement) || target.dataset.action !== "save") {
    return;
  }

  const row = target.closest("tr");
  if (!row) return;

  const username = row.dataset.username;
  const displayNameInput = row.querySelector("[data-field='display_name']");
  const roleSelect = row.querySelector("[data-field='role']");
  const enabledSelect = row.querySelector("[data-field='enabled']");

  if (!username || !displayNameInput || !roleSelect || !enabledSelect) {
    return;
  }

  target.setAttribute("disabled", "disabled");
  setStatus(adminStatus, `正在更新 ${username} ...`);

  try {
    await apiFetch(`/api/admin/users/${encodeURIComponent(username)}`, {
      method: "PATCH",
      body: {
        display_name: displayNameInput.value.trim(),
        role: roleSelect.value,
        enabled: enabledSelect.value === "true"
      }
    });
    await loadUsers();
    setStatus(adminStatus, `已更新用户 ${username}。`);
  } catch (err) {
    setStatus(adminStatus, `更新失败：${err.message}`, true);
  } finally {
    target.removeAttribute("disabled");
  }
});

async function boot() {
  const user = await initProtectedPage();
  state.currentUser = user;

  if (user.role !== "admin") {
    setStatus(adminStatus, "当前账号不是管理员，正在跳转到智能推荐页...", true);
    setTimeout(() => {
      window.location.href = "/recommend";
    }, 900);
    return;
  }

  setStatus(adminStatus, "正在加载管理员数据...");
  try {
    await loadUsers();
    setStatus(adminStatus, "管理员页面已就绪。", false);
  } catch (err) {
    setStatus(adminStatus, `加载失败：${err.message}`, true);
  }
}

boot();


