import { $, apiFetch, initPublicPage, notify, setStatus } from "./公共.js";

const loginTabBtn = $("loginTabBtn");
const registerTabBtn = $("registerTabBtn");
const loginPanel = $("loginPanel");
const registerPanel = $("registerPanel");
const loginForm = $("loginForm");
const registerForm = $("registerForm");
const loginStatus = $("loginStatus");
const registerStatus = $("registerStatus");
const strengthFill = $("passwordStrengthFill");
const strengthText = $("passwordStrengthText");
const authScene = $("authScene");
const forgotPasswordBtn = $("forgotPasswordBtn");
const loginSwitchToRegisterBtn = $("loginSwitchToRegisterBtn");
const loginFooterSwitchBtn = $("loginFooterSwitchBtn");
const registerBackToLoginBtn = $("registerBackToLoginBtn");
const registerFooterSwitchBtn = $("registerFooterSwitchBtn");

const loginUsernameInput = $("loginUsernameInput");
const loginPasswordInput = $("loginPasswordInput");
const registerUsernameInput = $("registerUsernameInput");
const registerDisplayNameInput = $("registerDisplayNameInput");
const registerPasswordInput = $("registerPasswordInput");
const confirmPasswordInput = $("confirmPasswordInput");
const AUTH_SUCCESS_REDIRECT = "/recommend";

const allTextInputs = [
  loginUsernameInput,
  loginPasswordInput,
  registerUsernameInput,
  registerDisplayNameInput,
  registerPasswordInput,
  confirmPasswordInput
].filter(Boolean);

const passwordInputs = [
  loginPasswordInput,
  registerPasswordInput,
  confirmPasswordInput
].filter(Boolean);

function validatePasswordRules(password) {
  const errors = [];
  if (password.length < 8) errors.push("长度至少 8 位");
  if (!/[a-z]/.test(password)) errors.push("包含小写字母");
  if (!/[A-Z]/.test(password)) errors.push("包含大写字母");
  if (!/[0-9]/.test(password)) errors.push("包含数字");
  if (!/[^A-Za-z0-9]/.test(password)) errors.push("包含符号");
  return errors;
}

function passwordStrength(password) {
  let score = 0;
  if ((password || "").length >= 8) score += 1;
  if ((password || "").length >= 12) score += 1;
  if (/[a-z]/.test(password)) score += 1;
  if (/[A-Z]/.test(password)) score += 1;
  if (/[0-9]/.test(password)) score += 1;
  if (/[^A-Za-z0-9]/.test(password)) score += 1;

  const bounded = Math.max(0, Math.min(6, score));
  let level = "弱";
  if (bounded >= 5) level = "强";
  else if (bounded >= 4) level = "较强";
  else if (bounded >= 3) level = "中";

  return {
    level,
    score: bounded,
    percent: Math.round((bounded / 6) * 100)
  };
}

function renderStrength(password) {
  if (!strengthFill || !strengthText) return;
  const result = passwordStrength(password);
  strengthFill.style.width = `${result.percent}%`;
  strengthText.textContent = `当前强度：${result.level}，${result.score}/6`;
}

function getActiveTab() {
  return loginPanel?.classList.contains("active") ? "login" : "register";
}

function switchTab(tab) {
  const showLogin = tab === "login";
  loginTabBtn?.classList.toggle("active", showLogin);
  registerTabBtn?.classList.toggle("active", !showLogin);
  loginTabBtn?.setAttribute("aria-selected", showLogin ? "true" : "false");
  registerTabBtn?.setAttribute("aria-selected", showLogin ? "false" : "true");
  loginPanel?.classList.toggle("active", showLogin);
  registerPanel?.classList.toggle("active", !showLogin);

  window.requestAnimationFrame(() => {
    if (showLogin) {
      loginUsernameInput?.focus();
    } else {
      registerUsernameInput?.focus();
    }
    syncSceneState();
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setSceneLook(clientX, clientY) {
  if (!authScene) return;
  if (
    authScene.classList.contains("is-password-mode") ||
    authScene.classList.contains("is-password-visible")
  ) {
    applySceneLookPreset();
    return;
  }
  const rect = authScene.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const offsetX = ((clientX - rect.left) / rect.width - 0.5) * 10;
  const offsetY = ((clientY - rect.top) / rect.height - 0.5) * 10;
  authScene.style.setProperty("--look-x", `${clamp(offsetX, -5, 5)}px`);
  authScene.style.setProperty("--look-y", `${clamp(offsetY, -4, 4)}px`);
}

function applySceneLookPreset() {
  if (!authScene) return;
  if (authScene.classList.contains("is-password-mode")) {
    authScene.style.setProperty("--look-x", "-6px");
    authScene.style.setProperty("--look-y", "-4px");
    return;
  }
  if (authScene.classList.contains("is-password-visible")) {
    authScene.style.setProperty("--look-x", "4px");
    authScene.style.setProperty("--look-y", "-2px");
    return;
  }
  authScene.style.setProperty("--look-x", "0px");
  authScene.style.setProperty("--look-y", "0px");
}

function resetSceneLook() {
  applySceneLookPreset();
}

function syncSceneState() {
  if (!authScene) return;
  const activeElement = document.activeElement;
  const isTyping = allTextInputs.includes(activeElement);
  const activePasswordInput = passwordInputs.find((input) => input === activeElement);
  const isPasswordFocused = Boolean(activePasswordInput);
  const isPasswordVisible = Boolean(activePasswordInput && activePasswordInput.type === "text");

  authScene.classList.toggle("is-typing", Boolean(isTyping && !isPasswordFocused));
  authScene.classList.toggle("is-password-mode", Boolean(isPasswordFocused && !isPasswordVisible));
  authScene.classList.toggle("is-password-visible", Boolean(isPasswordFocused && isPasswordVisible));
  applySceneLookPreset();
}

function wireScene() {
  if (!authScene) return;

  window.addEventListener("pointermove", (event) => {
    setSceneLook(event.clientX, event.clientY);
  });
  document.addEventListener("pointerleave", resetSceneLook);
  window.addEventListener("blur", resetSceneLook);

  allTextInputs.forEach((input) => {
    input.addEventListener("focus", syncSceneState);
    input.addEventListener("blur", () => {
      window.setTimeout(syncSceneState, 0);
    });
    input.addEventListener("input", syncSceneState);
  });

  syncSceneState();
}

function wirePasswordToggle() {
  const syncToggleState = (btn, input) => {
    const isRevealed = input.type === "text";
    btn.classList.toggle("is-revealed", isRevealed);
    btn.setAttribute("aria-label", isRevealed ? "隐藏密码" : "显示密码");
  };

  document.querySelectorAll("[data-toggle-target]").forEach((btn) => {
    if (!(btn instanceof HTMLButtonElement) || btn.dataset.bound === "1") return;
    btn.dataset.bound = "1";

    const targetId = btn.getAttribute("data-toggle-target");
    const input = targetId ? document.getElementById(targetId) : null;
    if (!(input instanceof HTMLInputElement)) return;

    syncToggleState(btn, input);
    btn.addEventListener("click", () => {
      input.type = input.type === "password" ? "text" : "password";
      syncToggleState(btn, input);
      syncSceneState();
    });
  });
}

function wireCapsLockHint() {
  document.querySelectorAll("input[type='password'][data-caps-tip]").forEach((input) => {
    if (!(input instanceof HTMLInputElement) || input.dataset.capsBound === "1") return;
    input.dataset.capsBound = "1";

    const tipId = input.getAttribute("data-caps-tip");
    const tip = tipId ? document.getElementById(tipId) : null;
    if (!(tip instanceof HTMLElement)) return;

    const onKeyEvent = (event) => {
      const isOn = Boolean(event.getModifierState && event.getModifierState("CapsLock"));
      tip.classList.toggle("show", isOn);
    };

    input.addEventListener("keydown", onKeyEvent);
    input.addEventListener("keyup", onKeyEvent);
    input.addEventListener("blur", () => tip.classList.remove("show"));
  });
}

function setSubmitDisabled(form, disabled) {
  const submitBtn = form?.querySelector("button[type='submit']");
  if (submitBtn instanceof HTMLButtonElement) {
    submitBtn.disabled = disabled;
  }
}

async function quickCheck() {
  try {
    await apiFetch("/api/auth/me");
    window.location.href = AUTH_SUCCESS_REDIRECT;
  } catch (_) {
    // User is not logged in yet.
  }
}

loginTabBtn?.addEventListener("click", () => switchTab("login"));
registerTabBtn?.addEventListener("click", () => switchTab("register"));
loginSwitchToRegisterBtn?.addEventListener("click", () => switchTab("register"));
loginFooterSwitchBtn?.addEventListener("click", () => switchTab("register"));
registerBackToLoginBtn?.addEventListener("click", () => switchTab("login"));
registerFooterSwitchBtn?.addEventListener("click", () => switchTab("login"));

forgotPasswordBtn?.addEventListener("click", () => {
  notify("当前版本暂不支持自助重置密码，请联系管理员处理。", "info", 4200);
});

loginForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  setSubmitDisabled(loginForm, true);
  setStatus(loginStatus, "正在登录...");

  try {
    await apiFetch("/api/auth/login", {
      method: "POST",
      body: {
        username: loginUsernameInput?.value.trim(),
        password: loginPasswordInput?.value || ""
      }
    });
    setStatus(loginStatus, "登录成功，正在进入系统...");
    notify("登录成功，正在进入推荐页。", "success", 1800);
    window.location.href = AUTH_SUCCESS_REDIRECT;
  } catch (error) {
    setStatus(loginStatus, `登录失败：${error.message}`, true);
  } finally {
    setSubmitDisabled(loginForm, false);
  }
});

registerForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  setSubmitDisabled(registerForm, true);
  setStatus(registerStatus, "正在注册...");

  const password = registerPasswordInput?.value || "";
  const confirmPassword = confirmPasswordInput?.value || "";

  if (password !== confirmPassword) {
    setStatus(registerStatus, "两次输入的密码不一致。", true);
    setSubmitDisabled(registerForm, false);
    return;
  }

  const passwordErrors = validatePasswordRules(password);
  if (passwordErrors.length) {
    setStatus(registerStatus, `密码不符合规则：${passwordErrors.join("、")}`, true);
    setSubmitDisabled(registerForm, false);
    return;
  }

  try {
    await apiFetch("/api/auth/register", {
      method: "POST",
      body: {
        username: registerUsernameInput?.value.trim(),
        display_name: registerDisplayNameInput?.value.trim(),
        password
      }
    });
    setStatus(registerStatus, "注册成功，正在进入系统...");
    notify("注册成功，系统已自动登录，正在进入推荐页。", "success", 2200);
    window.location.href = AUTH_SUCCESS_REDIRECT;
  } catch (error) {
    setStatus(registerStatus, `注册失败：${error.message}`, true);
  } finally {
    setSubmitDisabled(registerForm, false);
  }
});

registerPasswordInput?.addEventListener("input", () => {
  renderStrength(registerPasswordInput.value);
  syncSceneState();
});

confirmPasswordInput?.addEventListener("input", syncSceneState);
loginPasswordInput?.addEventListener("input", syncSceneState);

initPublicPage();
wirePasswordToggle();
wireCapsLockHint();
wireScene();
renderStrength(registerPasswordInput?.value || "");
switchTab(getActiveTab());
quickCheck();
