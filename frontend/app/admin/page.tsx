"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AdminShell } from "@/components/admin/AdminShell";
import { AdminKpiCards } from "@/components/admin/AdminKpiCards";
import { RoomTable } from "@/components/admin/RoomTable";
import { Button } from "@/components/ui/Button";
import { getAdminRooms } from "@/lib/api";
import type { Room } from "@/lib/types";

export default function AdminDashboardPage() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAdminRooms()
      .then(setRooms)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "불러오기 실패")
      )
      .finally(() => setLoading(false));
  }, []);

  const pendingCount = rooms.filter((r) => r.status === "pending_review").length;

  return (
    <AdminShell>
      <div className="space-y-6">
        {/* 헤더 */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">대시보드</h1>
            <p className="text-sm text-gray-500 mt-0.5">기숙사 퇴사 점검 현황</p>
          </div>
          <Link href="/admin/rooms">
            <Button variant="primary" size="sm">
              + 호실 등록
            </Button>
          </Link>
        </div>

        {/* 에러 */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* KPI 카드 */}
        {loading ? (
          <div className="grid grid-cols-3 lg:grid-cols-5 gap-3">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="h-24 bg-gray-100 rounded-2xl animate-pulse"
              />
            ))}
          </div>
        ) : (
          <AdminKpiCards rooms={rooms} />
        )}

        {/* 호실 목록 */}
        <div className="bg-white rounded-2xl border border-gray-100 p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-800">전체 호실</h2>
            <Link
              href="/admin/rooms"
              className="text-xs text-indigo-600 hover:underline"
            >
              관리 →
            </Link>
          </div>
          {loading ? (
            <div className="space-y-2">
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className="h-10 bg-gray-100 rounded-xl animate-pulse"
                />
              ))}
            </div>
          ) : (
            <RoomTable rooms={rooms.slice(0, 5)} />
          )}
          {rooms.length > 5 && (
            <p className="text-xs text-gray-400 text-center mt-3">
              외 {rooms.length - 5}개 호실 — 호실 관리에서 전체 확인
            </p>
          )}
        </div>

        {/* 빠른 링크 */}
        <div className="grid grid-cols-2 gap-3">
          <Link href="/admin/reviews">
            <div className="w-full bg-yellow-50 border border-yellow-200 rounded-2xl p-4 text-left hover:bg-yellow-100 transition-colors cursor-pointer">
              <span className="text-2xl">📋</span>
              <p className="text-sm font-semibold text-yellow-800 mt-1">
                점검 리뷰
              </p>
              <p className="text-xs text-yellow-600 mt-0.5">
                {loading ? "…" : `대기 ${pendingCount}건`}
              </p>
            </div>
          </Link>
          <Link href="/admin/rooms">
            <div className="w-full bg-indigo-50 border border-indigo-200 rounded-2xl p-4 text-left hover:bg-indigo-100 transition-colors cursor-pointer">
              <span className="text-2xl">🏢</span>
              <p className="text-sm font-semibold text-indigo-800 mt-1">
                호실 관리
              </p>
              <p className="text-xs text-indigo-600 mt-0.5">
                {loading ? "…" : `${rooms.length}개 등록됨`}
              </p>
            </div>
          </Link>
        </div>
      </div>
    </AdminShell>
  );
}
