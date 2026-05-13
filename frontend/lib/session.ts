/**
 * DormitorySession localStorage 유틸리티.
 *
 * - SSR 환경(window 없음)에서 에러가 나지 않도록 isBrowser() 체크 포함
 * - JSON parse 실패 시 세션 삭제 후 null 반환
 * - session key: "dormitory_session"
 */
import type { DormitorySession } from "./types";

const SESSION_KEY = "dormitory_session";

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

export function getDormitorySession(): DormitorySession | null {
  if (!isBrowser()) return null;
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as DormitorySession;
  } catch {
    // JSON parse 실패 → 손상된 데이터 삭제
    window.localStorage.removeItem(SESSION_KEY);
    return null;
  }
}

export function setDormitorySession(session: DormitorySession): void {
  if (!isBrowser()) return;
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

export function clearDormitorySession(): void {
  if (!isBrowser()) return;
  window.localStorage.removeItem(SESSION_KEY);
}

export function updateDormitorySession(
  partial: Partial<DormitorySession>
): void {
  const current = getDormitorySession();
  if (!current) return;
  setDormitorySession({ ...current, ...partial });
}

export function hasValidDormitorySession(): boolean {
  const s = getDormitorySession();
  return s !== null && typeof s.studentId === "number";
}

/** DormitorySession status 기반으로 이동할 경로를 반환합니다. */
export function statusToPath(status: string): string {
  switch (status) {
    case "ready":          return "/student/checkin";
    case "checked_in":     return "/student/checkout";
    case "pending_review":
    case "approved":
    case "rejected":       return "/student/result";
    default:               return "/student/checkin";
  }
}
