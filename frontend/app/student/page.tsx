"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { StudentShell } from "@/components/student/StudentShell";
import { getInspectionStatus } from "@/lib/api";
import { getDormitorySession, statusToPath, updateDormitorySession } from "@/lib/session";

export default function StudentFlowHubPage() {
  const router = useRouter();

  useEffect(() => {
    const session = getDormitorySession();
    if (!session) {
      router.replace("/student/login");
      return;
    }

    if (!session.inspectionId) {
      router.replace(statusToPath(session.status));
      return;
    }

    getInspectionStatus(session.inspectionId)
      .then((statusRes) => {
        updateDormitorySession({
          status: statusRes.status,
          adminFeedback: statusRes.admin_feedback ?? null,
        });
        router.replace(statusToPath(statusRes.status));
      })
      .catch(() => {
        router.replace(statusToPath(session.status));
      });
  }, [router]);

  return (
    <StudentShell title="학생 점검">
      <div className="flex min-h-[calc(100dvh-12rem)] flex-col items-center justify-center text-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-100 border-t-blue-600" />
        <p className="mt-4 text-base font-bold text-gray-950">진행 상태를 확인하고 있어요</p>
        <p className="mt-2 text-sm text-gray-500">필요한 단계로 바로 이동합니다.</p>
      </div>
    </StudentShell>
  );
}
