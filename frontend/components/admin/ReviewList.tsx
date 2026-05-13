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
      <div className="text-center py-12 text-gray-400 text-sm">
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
                "w-full text-left px-4 py-3 rounded-xl border transition-colors",
                selectedId === insp.id
                  ? "border-indigo-400 bg-indigo-50"
                  : "border-gray-100 bg-white hover:bg-gray-50",
              ].join(" ")}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold text-gray-900">
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
                      <span className="ml-2 text-orange-500">
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
