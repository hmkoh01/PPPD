"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AdminShell } from "@/components/admin/AdminShell";
import { RoomTable } from "@/components/admin/RoomTable";
import { Button } from "@/components/ui/Button";
import { getAdminRooms } from "@/lib/api";
import type { Room } from "@/lib/types";

export default function AdminRoomsPage() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAdminRooms()
      .then(setRooms)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "호실 목록을 불러오지 못했습니다.")
      )
      .finally(() => setLoading(false));
  }, []);

  return (
    <AdminShell>
      <div className="space-y-6">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl font-extrabold text-gray-950">호실 관리</h1>
            <p className="mt-1 text-sm text-gray-500">
              등록된 호실과 학생 상태를 확인합니다.
            </p>
          </div>
          <Link href="/admin/rooms/new">
            <Button type="button" variant="primary" size="sm">
              + 호실 등록
            </Button>
          </Link>
        </div>

        {error && (
          <div className="rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
            {error}
          </div>
        )}

        <div className="rounded-[24px] bg-white p-5 ring-1 ring-gray-100">
          <h2 className="mb-4 text-sm font-semibold text-gray-800">
            전체 호실{!loading && ` (${rooms.length}개)`}
          </h2>
          {loading ? (
            <div className="space-y-2">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-20 animate-pulse rounded-[20px] bg-gray-100" />
              ))}
            </div>
          ) : (
            <RoomTable rooms={rooms} />
          )}
        </div>
      </div>
    </AdminShell>
  );
}
