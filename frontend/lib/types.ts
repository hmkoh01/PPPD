/** backend API 응답에 대응하는 TypeScript 타입 정의. */

export type RoomStatus =
  | "ready"
  | "checked_in"
  | "needs_confirmation"
  | "pending_review"
  | "approved"
  | "rejected";

/** AI가 판정자가 아니라 확인 보조자라는 철학에 따른 의미 중심 상태값 */
export type IssueStatus =
  | "needs_confirmation"  // AI 감지 완료, 학생 확인 자료(근접 촬영+메모) 미제출
  | "evidence_submitted"  // 학생이 근접 촬영과 메모를 제출한 상태
  | "cleared";            // 추가 확인이 필요 없다고 판단된 상태

// ── 학생 ───────────────────────────────────────────────────

export interface Student {
  id: number;
  student_number: string;
  name: string;
  room_id: number | null;
}

// ── 이슈 ───────────────────────────────────────────────────

/** AI 감지 후보 유형 */
export type CandidateType =
  | "small_damage"           // 작은 손상/오염 후보
  | "large_object"           // 큰 객체/방치물 후보
  | "recapture_recommended"; // 이미지 전체 변화 — 재촬영 권장

export interface Issue {
  id: number;
  inspection_id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  status: IssueStatus;
  candidate_type?: CandidateType | string;
  student_note?: string | null;
  crop_image_path?: string | null;
  crop_image_url?: string | null;
  closeup_image_path?: string | null;
  closeup_image_url?: string | null;
  vlm_reason?: string | null;
}

// ── 점검 세션 ──────────────────────────────────────────────

export interface Inspection {
  id: number;
  room_id: number;
  student_id: number | null;
  status: RoomStatus;
  ref_image_path?: string | null;
  ref_image_url?: string | null;
  initial_image_path?: string | null;
  initial_image_url?: string | null;
  final_image_path?: string | null;
  final_image_url?: string | null;
  admin_feedback?: string | null;
  submitted_at?: string | null;
  reviewed_at?: string | null;
  issues?: Issue[];
}

export interface InspectionStatus {
  inspection_id: number;
  status: RoomStatus;
  admin_feedback?: string | null;
}

// ── 호실 ───────────────────────────────────────────────────

export interface Room {
  id: number;
  room_number: string;
  status: RoomStatus;
  student?: Student | null;
  latest_inspection?: Inspection | null;
  ref_image_url?: string | null;
}

// ── API 응답 ───────────────────────────────────────────────

export interface StudentVerifyResponse {
  student: Student;
  room: Room;
  inspection: Inspection | null;
  status: RoomStatus;
}

export interface InitialImageResponse {
  inspection_id: number;
  status: RoomStatus;
  initial_image_path: string;
  initial_image_url?: string | null;
}

export interface FinalImageResponse {
  inspection: Inspection;
  final_image_url?: string | null;
  issues: Issue[];
}

export interface CloseupResponse {
  issue: Issue;
  closeup_image_url?: string | null;
  result: "clean" | "suspicious";
  reason: string;
}

export interface AdminCreateRoomResponse {
  room: Room;
  student: Student;
  inspection: Inspection;
  ref_image_url?: string | null;
}

// ── localStorage 세션 ──────────────────────────────────────

export interface DormitorySession {
  studentId: number;
  studentNumber: string;
  studentName: string;
  roomId: number;
  roomNumber: string;
  inspectionId: number | null;
  status: RoomStatus;
  adminFeedback?: string | null;
  refImageUrl?: string | null;
  initialImageUrl?: string | null;
  finalImageUrl?: string | null;
}
