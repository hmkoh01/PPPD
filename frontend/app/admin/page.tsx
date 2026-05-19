"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { AdminShell } from "@/components/admin/AdminShell";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { getAdminInspections, getAdminRooms } from "@/lib/api";
import { getStatusLabel, getStatusTone } from "@/lib/status";
import type { Inspection, Room } from "@/lib/types";

function formatSubmittedAt(value: string | null | undefined): string {
  if (!value) return "제출일 없음";
  return new Date(value).toLocaleString("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function AdminDashboardPage() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [pendingInspections, setPendingInspections] = useState<Inspection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([getAdminRooms(), getAdminInspections("pending_review")])
      .then(([roomList, inspectionList]) => {
        setRooms(roomList);
        setPendingInspections(inspectionList);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "불러오기 실패")
      )
      .finally(() => setLoading(false));
  }, []);

  const roomsById = useMemo(() => {
    const map = new Map<number, Room>();
    rooms.forEach((room) => map.set(room.id, room));
    return map;
  }, [rooms]);

  const recentPending = useMemo(
    () =>
      [...pendingInspections]
        .sort((a, b) => {
          const left = a.submitted_at ? new Date(a.submitted_at).getTime() : 0;
          const right = b.submitted_at ? new Date(b.submitted_at).getTime() : 0;
          return right - left;
        })
        .slice(0, 5),
    [pendingInspections]
  );

  const rejectedCount = rooms.filter((room) => room.status === "rejected").length;
  const activeCount = rooms.filter((room) =>
    ["checked_in", "pending_review", "rejected"].includes(room.status)
  ).length;
  const approvedCount = rooms.filter((room) => room.status === "approved").length;

  return (
    <AdminShell>
      <div className="space-y-3">
        <section className="rounded-[24px] bg-white p-4 ring-1 ring-gray-100">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-extrabold text-gray-950">오늘 처리할 점검</h1>
              <p className="mt-2 text-4xl font-extrabold text-blue-600">
                {loading ? "..." : `검토 대기 ${pendingInspections.length}건`}
              </p>
              <p className="mt-2 text-sm font-medium text-gray-500">
                재촬영 요청 {loading ? "..." : rejectedCount}건
              </p>
            </div>
            <Link href="/admin/rooms/new">
              <Button variant="secondary" size="sm">
                + 호실 등록
              </Button>
            </Link>
          </div>

          <Link href="/admin/reviews" className="mt-4 block">
            <Button variant="primary" size="lg" fullWidth>
              검토하러 가기
            </Button>
          </Link>
        </section>

        {error && (
          <div className="rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
            {error}
          </div>
        )}

        <section className="rounded-[24px] bg-white p-4 ring-1 ring-gray-100">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-bold text-gray-950">최근 제출</h2>
            <Link href="/admin/reviews" className="text-xs font-bold text-blue-600">
              전체 보기
            </Link>
          </div>

          {loading ? (
            <div className="space-y-2">
              {[...Array(3)].map((_, index) => (
                <div key={index} className="h-16 animate-pulse rounded-2xl bg-gray-100" />
              ))}
            </div>
          ) : recentPending.length === 0 ? (
            <div className="rounded-2xl bg-gray-50 py-6 text-center">
              <p className="text-sm font-bold text-gray-700">아직 검토할 제출이 없어요</p>
              <p className="mt-1 text-xs text-gray-400">학생이 제출하면 여기에 표시돼요.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {recentPending.map((inspection) => {
                const room = roomsById.get(inspection.room_id);
                return (
                  <Link
                    key={inspection.id}
                    href="/admin/reviews"
                    className="block rounded-2xl bg-gray-50 px-4 py-3 transition-colors hover:bg-gray-100"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-base font-extrabold text-gray-950">
                          {room ? `${room.room_number}호` : `호실 #${inspection.room_id}`}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-gray-500">
                          {room?.student?.name ?? "학생 정보 없음"} · {formatSubmittedAt(inspection.submitted_at)}
                        </p>
                      </div>
                      <Badge tone={getStatusTone(inspection.status)}>
                        {getStatusLabel(inspection.status)}
                      </Badge>
                    </div>
                  </Link>
                );
              })}
            </div>
          )}
        </section>

        <section className="rounded-[24px] bg-white p-4 ring-1 ring-gray-100">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-bold text-gray-950">호실 현황</h2>
            <Link href="/admin/rooms" className="text-xs font-bold text-blue-600">
              호실 관리
            </Link>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded-2xl bg-gray-50 p-3">
              <p className="text-2xl font-extrabold text-gray-950">{loading ? "..." : rooms.length}</p>
              <p className="mt-1 text-xs font-semibold text-gray-500">전체</p>
            </div>
            <div className="rounded-2xl bg-blue-50 p-3">
              <p className="text-2xl font-extrabold text-blue-700">{loading ? "..." : activeCount}</p>
              <p className="mt-1 text-xs font-semibold text-blue-700">진행 중</p>
            </div>
            <div className="rounded-2xl bg-emerald-50 p-3">
              <p className="text-2xl font-extrabold text-emerald-700">{loading ? "..." : approvedCount}</p>
              <p className="mt-1 text-xs font-semibold text-emerald-700">완료</p>
            </div>
          </div>
        </section>
      </div>
    </AdminShell>
  );
}
