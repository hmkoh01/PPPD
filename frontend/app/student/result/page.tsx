"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { StudentShell } from "@/components/student/StudentShell";
import { StudentStepper } from "@/components/student/StudentStepper";
import { StudentStatusCard } from "@/components/student/StudentStatusCard";
import { Card } from "@/components/ui/Card";
import { getInspectionStatus } from "@/lib/api";
import { getDormitorySession, clearDormitorySession, updateDormitorySession } from "@/lib/session";
import type { DormitorySession, RoomStatus } from "@/lib/types";

const POLL_INTERVAL_MS = 12_000;

export default function StudentResultPage() {
  const router = useRouter();
  const [session, setSession] = useState<DormitorySession | null>(null);
  const [status, setStatus] = useState<RoomStatus | null>(null);
  const [adminFeedback, setAdminFeedback] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchStatus = useCallback(async (inspectionId: number) => {
    try {
      const res = await getInspectionStatus(inspectionId);
      const newStatus = res.status as RoomStatus;
      const newFeedback = res.admin_feedback ?? null;
      setStatus(newStatus);
      setAdminFeedback(newFeedback);
      setStatusError(null);
      updateDormitorySession({ status: newStatus, adminFeedback: newFeedback });
    } catch {
      setStatusError("상태를 새로 확인하지 못했어요.");
    }
  }, []);

  useEffect(() => {
    const s = getDormitorySession();
    if (!s) {
      router.replace("/student/login");
      return;
    }

    setSession(s);
    setStatus(s.status as RoomStatus);
    setAdminFeedback(s.adminFeedback ?? null);

    if (s.inspectionId) {
      fetchStatus(s.inspectionId);
      if (s.status === "pending_review") {
        const tick = () => {
          fetchStatus(s.inspectionId!).then(() => {
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

  if (!session || !status) return null;

  return (
    <StudentShell
      title={"\uc810\uac80 \uacb0\uacfc"}
      back={{ label: "\ucc98\uc74c\uc73c\ub85c", href: "/" }}
      trailing={`${session.roomNumber}\ud638`}
    >
      <StudentStepper currentStep={5} />

      <div className="flex min-h-[calc(100dvh-12rem)] flex-col gap-5">
      <StudentStatusCard status={status} adminFeedback={adminFeedback}>
        {status === "pending_review" && (
          <Button
            variant="secondary"
            size="sm"
            fullWidth
            onClick={() => session.inspectionId && fetchStatus(session.inspectionId)}
          >
            상태 새로고침
          </Button>
        )}
        {status === "rejected" && (
          <Link href="/student/checkout" className="block w-full">
            <Button variant="primary" size="md" fullWidth>
              재점검 시작
            </Button>
          </Link>
        )}
      </StudentStatusCard>

      {statusError && (
        <div className="flex items-center justify-between rounded-2xl bg-amber-50 px-4 py-3 text-xs text-amber-800 ring-1 ring-amber-100">
          <span>{statusError}</span>
          <button
            className="underline ml-2"
            onClick={() => session.inspectionId && fetchStatus(session.inspectionId)}
          >
            다시 확인
          </button>
        </div>
      )}

      <Card title="학생 정보">
        <div className="space-y-3 text-sm text-gray-700">
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
            <span className="font-medium">{session.roomNumber}</span>
          </div>
        </div>
      </Card>

        <div className="mt-auto">
          <Button variant="ghost" size="md" fullWidth onClick={handleLogout}>
            세션 종료
          </Button>
        </div>
      </div>
    </StudentShell>
  );
}
