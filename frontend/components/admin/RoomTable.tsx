"use client";

import { useState } from "react";
import type { Room } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { getStatusLabel, getStatusTone } from "@/lib/status";

interface RoomTableProps {
  rooms: Room[];
}

function RefThumb({ url }: { url: string | null | undefined }) {
  const [err, setErr] = useState(false);
  if (!url || err) {
    return (
      <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center text-gray-300 text-base">
        🖼️
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={url}
      alt="기준 사진"
      onError={() => setErr(true)}
      className="w-10 h-10 rounded-lg object-cover"
    />
  );
}

export function RoomTable({ rooms }: RoomTableProps) {
  if (rooms.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400 text-sm">
        등록된 호실이 없습니다.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="border-b border-gray-100 text-xs text-gray-500 uppercase tracking-wide">
            <th className="py-3 pr-3 w-10">사진</th>
            <th className="py-3 pr-4">호실</th>
            <th className="py-3 pr-4">학생</th>
            <th className="py-3 pr-4">학번</th>
            <th className="py-3">상태</th>
          </tr>
        </thead>
        <tbody>
          {rooms.map((room) => (
            <tr
              key={room.id}
              className="border-b border-gray-50 hover:bg-gray-50 transition-colors"
            >
              <td className="py-2 pr-3">
                <RefThumb url={room.ref_image_url} />
              </td>
              <td className="py-3 pr-4 font-semibold text-gray-900">
                {room.room_number}호
              </td>
              <td className="py-3 pr-4 text-gray-700">
                {room.student?.name ?? "—"}
              </td>
              <td className="py-3 pr-4 text-gray-500 font-mono text-xs">
                {room.student?.student_number ?? "—"}
              </td>
              <td className="py-3">
                <Badge tone={getStatusTone(room.status)}>
                  {getStatusLabel(room.status)}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
