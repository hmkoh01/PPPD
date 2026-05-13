"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper, statusToStep } from "@/components/student/StudentStepper";
import { StudentStatusCard } from "@/components/student/StudentStatusCard";
import { Card } from "@/components/ui/Card";
import { getInspectionStatus } from "@/lib/api";
import {
  getDormitorySession,
  clearDormitorySession,
  updateDormitorySession,
} from "@/lib/session";
import type { DormitorySession, RoomStatus } from "@/lib/types";

const POLL_INTERVAL_MS = 12_000; // 12초 간격 polling

export default function StudentResultPage() {
  const router = useRouter();
  const [session, setSession] = useState<DormitorySession | null>(null);
  const [status, setStatus] = useState<RoomStatus | null>(null);
  const [adminFeedback, setAdminFeedback] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchStatus = useCallback(
    async (inspectionId: number) => {
      try {
        const res = await getInspectionStatus(inspectionId);
        const newStatus = res.status as RoomStatus;
        const newFeedback = res.admin_feedback ?? null;
        setStatus(newStatus);
        setAdminFeedback(newFeedback);
        setStatusError(null);
        updateDormitorySession({ status: newStatus, adminFeedback: newFeedback });

        // 확정 상태(approved/rejected)면 polling 중단
        if (newStatus === "approved" || newStatus === "rejected") {
          if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        }
      } catch {
        setStatusError("상태 확인에 실패했습니다. 잠시 후 다시 시도하세요.");
      }
    },
    []
  );

  useEffect(() => {
    const s = getDormitorySession();
    if (!s) {
      router.replace("/student/login");
      return;
    }
    if (s.status === "ready") {
      router.replace("/student/checkin");
      return;
    }
    if (s.status === "checked_in") {
      router.replace("/student/checkout");
      return;
    }

    setSession(s);
    setStatus(s.status as RoomStatus);
    setAdminFeedback(s.adminFeedback ?? null);

    // 최신 상태를 API에서 가져오기
    if (s.inspectionId) {
      fetchStatus(s.inspectionId);

      // pending_review 상태일 때만 polling 시작
      if (s.status === "pending_review") {
        const tick = () => {
          fetchStatus(s.inspectionId!).then(() => {
            // status가 바뀌지 않았으면 계속 polling
            const current = getDormitorySession();
            if (current?.status === "pending_review") {
              pollTimerRef.current = setTimeout(tick, POLL_INTERVAL_MS);
            }
          });
        };
        pollTimerRef.current = setTimeout(tick, POLL_INTERVAL_MS);
      }
    }

    return () => {
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
  }, [router, fetchStatus]);

  const handleLogout = () => {
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    clearDormitorySession();
    router.push("/student");
  };

  const handleRefresh = () => {
    if (session?.inspectionId) fetchStatus(session.inspectionId);
  };

  if (!session || !status) return null;

  return (
    <StudentShell title="점검 결과" back={{ label: "처음으로", href: "/student" }}>
      <StudentStepper currentStep={statusToStep(status)} />

      <StudentStatusCard status={status} adminFeedback={adminFeedback}>
        {/* pending_review: 새로고침 버튼 */}
        {status === "pending_review" && (
          <Button variant="secondary" size="sm" fullWidth onClick={handleRefresh}>
            상태 새로고침
          </Button>
        )}
        {/* rejected: 재점검 시작 버튼 */}
        {status === "rejected" && (
          <Link href="/student/checkout" className="block w-full">
            <Button variant="primary" size="md" fullWidth>
              재점검 시작하기
            </Button>
          </Link>
        )}
        {/* approved: 완료 메시지 */}
        {status === "approved" && (
          <p className="text-xs text-green-700 font-medium text-center">
            수고하셨습니다! 퇴사 처리가 완료되었습니다.
          </p>
        )}
      </StudentStatusCard>

      {/* 상태 에러 알림 (non-blocking) */}
      {statusError && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl px-4 py-2 text-xs text-yellow-800 flex items-center justify-between">
          <span>{statusError}</span>
          <button className="underline ml-2" onClick={handleRefresh}>
            재시도
          </button>
        </div>
      )}

      {/* 점검 정보 */}
      <Card title="점검 정보">
        <div className="space-y-2 text-sm text-gray-700">
          <div className="flex justify-between">
            <span className="text-gray-500">이름</span>
            <span className="font-medium">{session.studentName}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">학번</span>
            <span className="font-medium font-mono">{session.studentNumber}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">호실</span>
            <span className="font-medium">{session.roomNumber}호</span>
          </div>
          {status === "pending_review" && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">자동 새로고침</span>
              <span className="text-gray-400">
                {POLL_INTERVAL_MS / 1000}초 간격
              </span>
            </div>
          )}
        </div>
      </Card>

      <Button variant="ghost" size="md" fullWidth onClick={handleLogout}>
        로그아웃
      </Button>
    </StudentShell>
  );
}
