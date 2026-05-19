"use client";

import { useState } from "react";
import type { Room } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { getStatusLabel, getStatusTone } from "@/lib/status";
import { resolveImageUrl } from "@/lib/image";

interface RoomTableProps {
  rooms: Room[];
}

function RefThumb({ url }: { url: string | null | undefined }) {
  const [err, setErr] = useState(false);
  const imageUrl = resolveImageUrl(url);
  if (!imageUrl || err) {
    return (
      <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gray-100 text-xs text-gray-300">
        사진
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={imageUrl}
      alt="기준 사진"
      onError={() => setErr(true)}
      className="h-14 w-14 shrink-0 rounded-2xl object-cover"
    />
  );
}

export function RoomTable({ rooms }: RoomTableProps) {
  if (rooms.length === 0) {
    return (
      <div className="rounded-[24px] bg-gray-50 py-12 text-center text-sm text-gray-400">
        등록된 호실이 없습니다.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {rooms.map((room) => (
        <div
          key={room.id}
          className="flex items-center gap-3 rounded-[20px] bg-gray-50 p-3"
        >
          <RefThumb url={room.ref_image_url} />
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="text-base font-extrabold text-gray-950">
                  {room.room_number}호
                </p>
                <p className="mt-0.5 truncate text-sm text-gray-600">
                  {room.student?.name ?? "학생 미등록"}
                </p>
                <p className="mt-0.5 truncate font-mono text-xs text-gray-400">
                  {room.student?.student_number ?? "-"}
                </p>
                <p className="mt-1 text-xs font-medium text-gray-400">
                  기준사진 {room.ref_image_url ? "등록" : "미등록"}
                </p>
              </div>
              <Badge tone={getStatusTone(room.status)}>
                {getStatusLabel(room.status)}
              </Badge>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
