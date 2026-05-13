import type { RoomStatus, IssueStatus } from "./types";

/** Inspection / Room 상태 → 한국어 라벨 */
export function getStatusLabel(status: RoomStatus): string {
  const labels: Record<RoomStatus, string> = {
    ready:          "등록 완료",
    checked_in:     "입사 사진 완료",
    pending_review: "관리자 검토 중",
    approved:       "승인 완료",
    rejected:       "재점검 요청",
  };
  return labels[status] ?? status;
}

/** Inspection / Room 상태 → UI 톤 (Tailwind 색 이름 기준) */
export type Tone = "gray" | "yellow" | "blue" | "green" | "red" | "orange";

export function getStatusTone(status: RoomStatus): Tone {
  const tones: Record<RoomStatus, Tone> = {
    ready:          "gray",
    checked_in:     "blue",
    pending_review: "yellow",
    approved:       "green",
    rejected:       "red",
  };
  return tones[status] ?? "gray";
}

/** Issue 상태 → 한국어 라벨 */
export function getIssueStatusLabel(status: IssueStatus): string {
  const labels: Record<IssueStatus, string> = {
    red:    "촬영 필요",
    orange: "확인 필요",
    green:  "이상 없음",
  };
  return labels[status] ?? status;
}

/** Issue 상태 → UI 톤 */
export function getIssueStatusTone(status: IssueStatus): Tone {
  const tones: Record<IssueStatus, Tone> = {
    red:    "red",
    orange: "orange",
    green:  "green",
  };
  return tones[status] ?? "gray";
}
