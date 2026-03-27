import { apiFetch } from "./公共.js";

export function fetchDefaultEnv() {
  return apiFetch("/api/default-env");
}

export function fetchRecommend(envPayload) {
  return apiFetch("/api/recommend", {
    method: "POST",
    body: { env: envPayload }
  });
}

export function fetchCropVisuals(payload) {
  return apiFetch("/api/crop-visuals", {
    method: "POST",
    body: payload
  });
}
