"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { verifyStudent } from "@/lib/api";
import { setDormitorySession, statusToPath } from "@/lib/session";
import type { DormitorySession } from "@/lib/types";

/** API 에러 메시지를 사용자 친화적 메시지로 변환합니다. */
function toUserMessage(err: unknown): string {
  if (!(err instanceof Error)) return "인증에 실패했습니다.";
  const msg = err.message;
  if (msg.includes("연결할 수 없")) return "서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.";
  if (msg.includes("배정") || msg.includes("일치")) return msg;
  return "일치하는 배정 정보를 찾을 수 없습니다. 학번과 이름을 확인해 주세요.";
}

export default function StudentLoginPage() {
  const router = useRouter();
  const [studentNumber, setStudentNumber] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentNumber.trim() || !name.trim()) {
      setError("학번과 이름을 모두 입력해 주세요.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await verifyStudent({
        student_number: studentNumber.trim(),
        name: name.trim(),
      });

      const session: DormitorySession = {
        studentId:      res.student.id,
        studentNumber:  res.student.student_number,
        studentName:    res.student.name,
        roomId:         res.room.id,
        roomNumber:     res.room.room_number,
        inspectionId:   res.inspection?.id ?? null,
        status:         res.status,
        adminFeedback:  res.inspection?.admin_feedback ?? null,
        refImageUrl:    res.inspection?.ref_image_url ?? null,
        initialImageUrl: res.inspection?.initial_image_url ?? null,
        finalImageUrl:  res.inspection?.final_image_url ?? null,
      };
      setDormitorySession(session);

      router.push(statusToPath(res.status));
    } catch (err) {
      setError(toUserMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <StudentShell title="학생 인증" back={{ label: "처음으로", href: "/" }}>
      <StudentStepper currentStep={1} />

      <div className="flex min-h-[calc(100dvh-12rem)] flex-col py-2">
        <div className="mb-8">
          <h2 className="text-3xl font-extrabold text-gray-950">배정 호실 확인</h2>
          <p className="mt-2 text-sm leading-relaxed text-gray-500">
            관리자에게 등록된 학번과 이름을 입력하세요.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-1 flex-col space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              학번
            </label>
            <input
              type="text"
              value={studentNumber}
              onChange={(e) => setStudentNumber(e.target.value)}
              placeholder="예) 20240000"
              className="h-14 w-full rounded-2xl bg-white px-4 text-base ring-1 ring-gray-100 transition focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
              autoComplete="off"
              inputMode="numeric"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              이름
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="예) 홍길동"
              className="h-14 w-full rounded-2xl bg-white px-4 text-base ring-1 ring-gray-100 transition focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
              autoComplete="off"
            />
          </div>

          {error && (
            <div className="rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
              {error}
            </div>
          )}

          <div className="mt-auto pt-4">
            <Button
              type="submit"
              variant="primary"
              size="lg"
              fullWidth
              disabled={loading}
            >
              {loading ? "확인 중…" : "시작하기"}
            </Button>
          </div>
        </form>

        <p className="mt-4 text-xs text-gray-400 text-center">
          배정 정보가 없다면 관리자에게 문의하세요.
        </p>
      </div>
    </StudentShell>
  );
}
