import type { Inspection, Room } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { getStatusLabel, getStatusTone } from "@/lib/status";

interface ReviewListProps {
  inspections: Inspection[];
  roomsById: Map<number, Room>;
  onSelect: (id: number) => void;
  selectedId?: number | null;
}

export function ReviewList({
  inspections,
  roomsById,
  onSelect,
  selectedId,
}: ReviewListProps) {
  if (inspections.length === 0) {
    return (
      <div className="rounded-[24px] bg-white py-12 text-center text-sm text-gray-400 ring-1 ring-gray-100">
        검토할 점검이 없습니다.
      </div>
    );
  }

  return (
    <ul className="space-y-2">
      {inspections.map((insp) => {
        const room = roomsById.get(insp.room_id);
        return (
          <li key={insp.id}>
            <button
              onClick={() => onSelect(insp.id)}
              className={[
                "w-full rounded-[20px] px-4 py-4 text-left transition-colors",
                selectedId === insp.id
                  ? "bg-blue-50 ring-2 ring-blue-200"
                  : "bg-white ring-1 ring-gray-100 hover:bg-gray-50",
              ].join(" ")}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-lg font-extrabold text-gray-950">
                      {room ? `${room.room_number}호` : `호실 #${insp.room_id}`}
                    </p>
                    {room?.student && (
                      <span className="text-xs text-gray-500">
                        {room.student.name} ({room.student.student_number})
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {insp.submitted_at
                      ? new Date(insp.submitted_at).toLocaleString("ko-KR")
                      : "제출일 없음"}
                    {insp.issues && insp.issues.length > 0 && (
                      <span className="ml-2 rounded-full bg-amber-50 px-2 py-0.5 text-amber-700">
                        이슈 {insp.issues.length}건
                      </span>
                    )}
                  </p>
                </div>
                <Badge tone={getStatusTone(insp.status)}>
                  {getStatusLabel(insp.status)}
                </Badge>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
