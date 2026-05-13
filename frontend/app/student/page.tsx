"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { StudentShell } from "@/components/student/StudentShell";
import { getInspectionStatus } from "@/lib/api";
import {
  getDormitorySession,
  updateDormitorySession,
  clearDormitorySession,
  statusToPath,
} from "@/lib/session";

type PageState = "idle" | "checking" | "no_session";

export default function StudentHomePage() {
  const router = useRouter();
  const [pageState, setPageState] = useState<PageState>("idle");
  const [warning, setWarning] = useState<string | null>(null);

  useEffect(() => {
    const session = getDormitorySession();
    if (!session || !session.inspectionId) {
      setPageState("no_session");
      return;
    }

    // 세션이 있으면 최신 상태를 API에서 확인
    setPageState("checking");
    getInspectionStatus(session.inspectionId)
      .then((statusRes) => {
        updateDormitorySession({
          status: statusRes.status,
          adminFeedback: statusRes.admin_feedback ?? null,
        });
        router.replace(statusToPath(statusRes.status));
      })
      .catch(() => {
        // API 실패 시 세션의 기존 status로 fallback
        setWarning("서버 연결에 실패했습니다. 이전 상태로 이동합니다.");
        setTimeout(() => {
          router.replace(statusToPath(session.status));
        }, 1500);
      });
  }, [router]);

  // 세션 없음 → 홈 화면 표시
  if (pageState === "no_session") {
    return (
      <StudentShell title="학생 점검">
        <div className="flex flex-col items-center text-center gap-6 py-8">
          <div className="space-y-2">
            <div className="text-5xl">📱</div>
            <h2 className="text-xl font-bold text-gray-900">퇴사 점검을 시작합니다</h2>
            <p className="text-sm text-gray-500 leading-relaxed">
              학번과 이름을 입력하면 배정된 호실을<br />
              자동으로 확인할 수 있습니다.
            </p>
          </div>

          <div className="w-full space-y-3">
            <Link href="/student/login" className="block">
              <Button variant="primary" size="lg" fullWidth>
                점검 시작하기
              </Button>
            </Link>
            <Link href="/" className="block">
              <Button variant="ghost" size="md" fullWidth>
                ← 처음으로
              </Button>
            </Link>
          </div>

          <div className="w-full bg-blue-50 rounded-2xl p-4 text-left space-y-1.5">
            <p className="text-xs font-semibold text-blue-700">점검 순서</p>
            {[
              "학번과 이름으로 인증",
              "입사 때 사진 촬영",
              "퇴사 전 사진 촬영 + AI 분석",
              "차이 영역 클로즈업 촬영",
              "최종 제출 → 관리자 검토",
            ].map((step, i) => (
              <p key={i} className="text-xs text-blue-700">
                {i + 1}. {step}
              </p>
            ))}
          </div>
        </div>
      </StudentShell>
    );
  }

  // 세션 확인 중 (loading) or warning
  return (
    <StudentShell title="학생 점검">
      <div className="flex flex-col items-center gap-4 py-16">
        {warning ? (
          <>
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl px-4 py-3 text-sm text-yellow-800 text-center">
              {warning}
            </div>
            <p className="text-xs text-gray-400">잠시 후 이동합니다…</p>
          </>
        ) : (
          <>
            <div className="w-8 h-8 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin" />
            <p className="text-sm text-gray-500">상태 확인 중…</p>
            <button
              className="text-xs text-gray-400 underline mt-2"
              onClick={() => {
                clearDormitorySession();
                setPageState("no_session");
              }}
            >
              취소하고 처음으로
            </button>
          </>
        )}
      </div>
    </StudentShell>
  );
}
