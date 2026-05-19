import type { RoomStatus, IssueStatus } from "./types";

/** Inspection / Room status labels. */
export function getStatusLabel(status: RoomStatus): string {
  const labels: Record<RoomStatus, string> = {
    ready: "등록 완료",
    checked_in: "입사 사진 완료",
    needs_confirmation: "확인 자료 제출 대기",
    pending_review: "관리자 검토 중",
    approved: "승인 완료",
    rejected: "재점검 요청",
  };
  return labels[status] ?? status;
}

/** Inspection / Room status UI tone. */
export type Tone = "gray" | "yellow" | "blue" | "green" | "red" | "orange";

export function getStatusTone(status: RoomStatus): Tone {
  const tones: Record<RoomStatus, Tone> = {
    ready: "gray",
    checked_in: "blue",
    needs_confirmation: "orange",
    pending_review: "yellow",
    approved: "green",
    rejected: "red",
  };
  return tones[status] ?? "gray";
}

/**
 * Issue status labels.
 * 중립적 표현 원칙: "손상", "파손" 같은 단정적 표현을 사용하지 않습니다.
 */
export function getIssueStatusLabel(status: IssueStatus): string {
  const labels: Record<IssueStatus, string> = {
    needs_confirmation: "확인 필요",
    evidence_submitted: "자료 제출됨",
    cleared: "이상 없음",
  };
  return labels[status] ?? status;
}

/** Issue status UI tone. */
export function getIssueStatusTone(status: IssueStatus): Tone {
  const tones: Record<IssueStatus, Tone> = {
    needs_confirmation: "yellow",
    evidence_submitted: "blue",
    cleared: "green",
  };
  return tones[status] ?? "gray";
}
