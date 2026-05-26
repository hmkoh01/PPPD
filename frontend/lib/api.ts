/**
 * FastAPI 백엔드 API client.
 * NEXT_PUBLIC_API_BASE_URL 환경변수를 기반으로 fetch 요청을 보냅니다.
 */
import type {
  AdminCreateRoomResponse,
  AlignmentCheckResponse,
  CloseupResponse,
  FinalImageResponse,
  InitialImageResponse,
  Inspection,
  InspectionStatus,
  Issue,
  Room,
  StudentVerifyResponse,
} from "./types";

const configuredBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "";

function isLocalhostUrl(url: string): boolean {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(url);
}

function shouldUseSameOrigin(baseUrl: string): boolean {
  if (!baseUrl) return true;
  if (typeof window === "undefined") return false;
  if (!isLocalhostUrl(baseUrl)) return false;

  const host = window.location.hostname;
  return host !== "localhost" && host !== "127.0.0.1";
}

function apiUrl(path: string): string {
  return shouldUseSameOrigin(configuredBaseUrl)
    ? path
    : `${configuredBaseUrl}${path}`;
}

// ── 기본 fetch 래퍼 ────────────────────────────────────────

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = apiUrl(path);

  // FormData 를 보낼 때는 Content-Type 을 수동으로 지정하지 않는다
  // (브라우저가 boundary 포함한 multipart/form-data 를 자동으로 설정)
  const isFormData = options.body instanceof FormData;
  const headers: Record<string, string> = isFormData
    ? {}
    : { "Content-Type": "application/json" };

  let res: Response;
  try {
    res = await fetch(url, {
      ...options,
      headers: { ...headers, ...(options.headers as Record<string, string>) },
    });
  } catch {
    throw new Error("서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요.");
  }

  if (!res.ok) {
    let message = `오류 ${res.status}`;
    try {
      const text = await res.text();
      try {
        const body = JSON.parse(text) as Record<string, unknown>;
        if (typeof body?.detail === "string") {
          message = body.detail;
        } else if (Array.isArray(body?.detail)) {
          // FastAPI validation error: [{loc, msg, type}]
          message = (body.detail as Array<{ loc?: string[]; msg: string }>)
            .map((e) => {
              const loc = e.loc?.at(-1) ?? "입력값";
              return `${loc}: ${e.msg}`;
            })
            .join(" / ");
        } else if (typeof body?.message === "string") {
          message = body.message;
        } else if (text) {
          message = text;
        }
      } catch {
        if (text) message = text;
      }
    } catch {
      message = res.statusText || message;
    }
    throw new Error(message);
  }

  return res.json() as Promise<T>;
}

// ── 학생 ───────────────────────────────────────────────────

export function verifyStudent(payload: {
  student_number: string;
  name: string;
}): Promise<StudentVerifyResponse> {
  return apiFetch("/api/students/verify", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ── 관리자 — 호실 ──────────────────────────────────────────

export function getAdminRooms(): Promise<Room[]> {
  return apiFetch("/api/admin/rooms");
}

export function createAdminRoom(formData: FormData): Promise<AdminCreateRoomResponse> {
  return apiFetch("/api/admin/rooms", {
    method: "POST",
    body: formData,
  });
}

export function getAdminRoom(roomId: number): Promise<Room> {
  return apiFetch(`/api/admin/rooms/${roomId}`);
}

// ── 관리자 — 점검 ──────────────────────────────────────────

export function getAdminInspections(statusFilter?: string): Promise<Inspection[]> {
  const qs = statusFilter ? `?status_filter=${statusFilter}` : "";
  return apiFetch(`/api/admin/inspections${qs}`);
}

export function getAdminInspection(id: number): Promise<Inspection> {
  return apiFetch(`/api/admin/inspections/${id}`);
}

export function approveInspection(
  id: number,
  payload?: { admin_feedback?: string }
): Promise<Inspection> {
  return apiFetch(`/api/admin/inspections/${id}/approve`, {
    method: "PATCH",
    body: JSON.stringify(payload ?? {}),
  });
}

export function rejectInspection(
  id: number,
  payload: { admin_feedback: string }
): Promise<Inspection> {
  return apiFetch(`/api/admin/inspections/${id}/reject`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

// ── 학생 — 점검 ────────────────────────────────────────────

export function getInspection(id: number): Promise<Inspection> {
  return apiFetch(`/api/inspections/${id}`);
}

export function getInspectionStatus(id: number): Promise<InspectionStatus> {
  return apiFetch(`/api/inspections/${id}/status`);
}

function _buildImageForm(fileOrBlob: File | Blob): FormData {
  const form = new FormData();
  form.append("file", fileOrBlob);
  return form;
}

export function uploadInitialImage(
  inspectionId: number,
  fileOrBlob: File | Blob
): Promise<InitialImageResponse> {
  return apiFetch(`/api/inspections/${inspectionId}/initial-image`, {
    method: "POST",
    body: _buildImageForm(fileOrBlob),
  });
}

export function uploadFinalImage(
  inspectionId: number,
  fileOrBlob: File | Blob
): Promise<FinalImageResponse> {
  return apiFetch(`/api/inspections/${inspectionId}/final-image`, {
    method: "POST",
    body: _buildImageForm(fileOrBlob),
  });
}

export function checkImageAlignment(
  inspectionId: number,
  fileOrBlob: File | Blob,
  mode: "checkin" | "checkout",
  preview = false,
): Promise<AlignmentCheckResponse> {
  const qs = `mode=${mode}${preview ? "&preview=true" : ""}`;
  return apiFetch(`/api/inspections/${inspectionId}/alignment-check?${qs}`, {
    method: "POST",
    body: _buildImageForm(fileOrBlob),
  });
}

export function submitInspection(inspectionId: number): Promise<InspectionStatus> {
  return apiFetch(`/api/inspections/${inspectionId}/submit`, {
    method: "POST",
  });
}

// ── 학생 — 이슈 ────────────────────────────────────────────

export function uploadIssueCloseup(
  issueId: number,
  fileOrBlob: File | Blob,
  studentNote?: string,
): Promise<CloseupResponse> {
  const form = new FormData();
  form.append("file", fileOrBlob);
  if (studentNote?.trim()) {
    form.append("student_note", studentNote.trim());
  }
  return apiFetch(`/api/issues/${issueId}/closeup`, {
    method: "POST",
    body: form,
  });
}

export function retakeIssue(issueId: number): Promise<Issue> {
  return apiFetch(`/api/issues/${issueId}/retake`, { method: "PATCH" });
}

export function clearIssue(issueId: number, studentNote?: string): Promise<Issue> {
  return apiFetch(`/api/issues/${issueId}/clear`, {
    method: "PATCH",
    body: JSON.stringify({ student_note: studentNote?.trim() || null }),
  });
}
