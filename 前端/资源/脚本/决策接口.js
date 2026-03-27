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

export function fetchHomeSummary() {
  return apiFetch("/api/home-summary");
}

export function fetchCompetitionOverview() {
  return apiFetch("/api/competition-overview");
}

export function fetchAssistantAnswer(questionId, crop, questionText = "") {
  return apiFetch("/api/assistant-answer", {
    method: "POST",
    body: {
      question_id: questionId || null,
      crop: crop || null,
      question_text: questionText || null
    }
  });
}

export function fetchStoreSummary() {
  return apiFetch("/api/store-summary");
}

export function fetchCommunitySummary() {
  return apiFetch("/api/community-summary");
}

export function fetchProfileSummary() {
  return apiFetch("/api/profile-summary");
}

export function fetchRecommendHistory() {
  return apiFetch("/api/recommend-history");
}
